"""
Discrete-event simulation of the matching system with two applicant classes, 
one employer class and weighted matching policies.
"""

import heapq
from collections import deque

import numpy as np

PRIORITY_1 = float("inf")
PRIORITY_2 = 0.0
UNIFORM = 1.0


class Simulation:

    def __init__(self, model, policy_A, warm_up, run_length, seed=None):
        self.model = model
        self.policy_A = float(policy_A)
        self.warm_up = warm_up
        self.run_length = run_length
        self.termination_time = warm_up + run_length
        self.rng = np.random.default_rng(seed)

        # clock and event calendar
        self.clock = 0.0
        self._event_sequence = 0
        self.event_calendar = []

        # queues
        self.queue_A1 = deque()
        self.queue_A2 = deque()
        self.queue_E = deque()

        # participant records
        self._next_participant_id = 1
        self.participant_class = {}
        self.queue_entry_time = {}
        self.status = {}

        self._warm_up_done = (warm_up == 0)
        self.initial_system_size = 0
        self._reset_statistics()

        self.maximum_queue_lengths = {"A1": 0, "A2": 0, "E": 0}
        self.initialise()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _reset_statistics(self):
        self.arrivals = {"A1": 0, "A2": 0, "E": 0}
        self.admitted = {"A1": 0, "A2": 0, "E": 0}
        self.matches = {"A1": 0, "A2": 0}
        self.abandonments = {"A1": 0, "A2": 0, "E": 0}
        self.rejections = {"A1": 0, "A2": 0, "E": 0}

        # A scan occurs when an arrival finds the opposite queue non-empty and checks for acceptable partners.
        self.scans = 0
        self.scans_successful = 0

        # Separate waiting times are recorded for matched and abandoned participants, then combined for Little's Law.
        self.wait_matched_total = {"A1": 0.0, "A2": 0.0, "E": 0.0}
        self.wait_matched_count = {"A1": 0, "A2": 0, "E": 0}
        self.wait_abandoned_total = {"A1": 0.0, "A2": 0.0, "E": 0.0}
        self.wait_abandoned_count = {"A1": 0, "A2": 0, "E": 0}

        self.queue_area = {"A1": 0.0, "A2": 0.0, "E": 0.0}

    def initialise(self):
        self.schedule_event(self.draw_interarrival("A1"), "arrival_A1")
        self.schedule_event(self.draw_interarrival("A2"), "arrival_A2")
        self.schedule_event(self.draw_interarrival("E"), "arrival_E")
        self.schedule_event(self.termination_time, "termination")

    def schedule_event(self, event_time, event_type, participant_id=None):
        self._event_sequence += 1
        heapq.heappush(self.event_calendar,
                       (event_time, self._event_sequence, event_type, participant_id))

    # ------------------------------------------------------------------
    # Random draws
    # ------------------------------------------------------------------
    def draw_interarrival(self, participant_class):
        rate = {"A1": self.model.lambda_1,
                "A2": self.model.lambda_2,
                "E": self.model.lambda_e}[participant_class]
        if rate <= 0:
            return float("inf")
        return self.rng.exponential(1.0 / rate)

    def draw_patience(self, participant_class):
        rate = {"A1": self.model.theta_1,
                "A2": self.model.theta_2,
                "E": self.model.theta_e}[participant_class]
        if rate <= 0:
            return float("inf")
        return self.rng.exponential(1.0 / rate)

    def draw_acceptable(self, queue_size, q):
        """Number of waiting individuals acceptable to the arriving user."""
        if queue_size == 0 or q <= 0:
            return 0
        if q >= 1:
            return queue_size
        return int(self.rng.binomial(queue_size, q))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        while self.event_calendar:
            event_time, _, event_type, participant_id = heapq.heappop(self.event_calendar)

            self._cross_warm_up(self.clock, event_time)
            self._accumulate_queue_area(self.clock, event_time)
            self.clock = event_time

            if event_type == "termination":
                break
            elif event_type == "arrival_A1":
                self.arrival_applicant("A1")
            elif event_type == "arrival_A2":
                self.arrival_applicant("A2")
            elif event_type == "arrival_E":
                self.arrival_employer()
            elif event_type == "abandonment":
                self.abandon(participant_id)

            self._update_maximum_queue_lengths()

        return self.results()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def arrival_applicant(self, applicant_class):
        self.schedule_event(self.clock + self.draw_interarrival(applicant_class),
                            f"arrival_{applicant_class}")
        if self._observing():
            self.arrivals[applicant_class] += 1

        applicant_id = self.new_participant(applicant_class)
        q = self.model.q_1 if applicant_class == "A1" else self.model.q_2

        if self.queue_E:
            acceptable = self.draw_acceptable(len(self.queue_E), q)
            if self._observing():
                self.scans += 1
            if acceptable > 0:
                if self._observing():
                    self.scans_successful += 1
                employer_id = self.remove_random(self.queue_E)
                self.complete_match(applicant_class, waiting_id=employer_id,
                                    arriving_id=applicant_id)
                return

        self.attempt_admission(applicant_id, applicant_class)

    def arrival_employer(self):
        self.schedule_event(self.clock + self.draw_interarrival("E"), "arrival_E")
        if self._observing():
            self.arrivals["E"] += 1

        employer_id = self.new_participant("E")
        x1, x2 = len(self.queue_A1), len(self.queue_A2)

        if x1 > 0 or x2 > 0:
            acceptable_1 = self.draw_acceptable(x1, self.model.q_1)
            acceptable_2 = self.draw_acceptable(x2, self.model.q_2)
            if self._observing():
                self.scans += 1
            if acceptable_1 + acceptable_2 > 0:
                if self._observing():
                    self.scans_successful += 1
                selected_class = self.select_class(acceptable_1, acceptable_2)
                queue = self.queue_A1 if selected_class == "A1" else self.queue_A2
                applicant_id = self.remove_random(queue)
                self.complete_match(selected_class, waiting_id=applicant_id,
                                    arriving_id=employer_id)
                return

        self.attempt_admission(employer_id, "E")

    def select_class(self, acceptable_1, acceptable_2):
        """Chooses between acceptable candidates using policy weight A."""
        if acceptable_1 == 0:
            return "A2"
        if acceptable_2 == 0:
            return "A1"
        if self.policy_A == float("inf"):
            return "A1"
        if self.policy_A <= 0.0:
            return "A2"
        weight_1 = self.policy_A * acceptable_1
        probability_1 = weight_1 / (weight_1 + acceptable_2)
        return "A1" if self.rng.random() < probability_1 else "A2"

    def attempt_admission(self, participant_id, participant_class):
        queue, capacity = self.queue_and_capacity(participant_class)
        if len(queue) >= capacity:
            self.status[participant_id] = "rejected"
            if self._observing():
                self.rejections[participant_class] += 1
            return

        self.status[participant_id] = "waiting"
        self.queue_entry_time[participant_id] = self.clock
        queue.append(participant_id)
        if self._observing():
            self.admitted[participant_class] += 1

        patience = self.draw_patience(participant_class)
        if patience != float("inf"):
            self.schedule_event(self.clock + patience, "abandonment", participant_id)

    def abandon(self, participant_id):
        # A scheduled abandonment is ignored if the participant has since matched.
        if self.status.get(participant_id) != "waiting":
            return

        participant_class = self.participant_class[participant_id]
        queue, _ = self.queue_and_capacity(participant_class)
        queue.remove(participant_id)
        self.status[participant_id] = "abandoned"
        if self._observing():
            wait = self.clock - self.queue_entry_time[participant_id]
            self.abandonments[participant_class] += 1
            self.wait_abandoned_total[participant_class] += wait
            self.wait_abandoned_count[participant_class] += 1

    def complete_match(self, applicant_class, waiting_id, arriving_id):
        self.status[waiting_id] = "matched"
        self.status[arriving_id] = "matched"
        if self._observing():
            self.matches[applicant_class] += 1
            waiting_class = self.participant_class[waiting_id]
            wait = self.clock - self.queue_entry_time[waiting_id]
            self.wait_matched_total[waiting_class] += wait
            self.wait_matched_count[waiting_class] += 1

    # ------------------------------------------------------------------
    # Tracking Helpers
    # ------------------------------------------------------------------
    def remove_random(self, queue):
        """Remove and return a uniformly chosen member of `queue`."""
        index = int(self.rng.integers(len(queue)))
        participant_id = queue[index]
        del queue[index]
        return participant_id

    def new_participant(self, participant_class):
        participant_id = self._next_participant_id
        self._next_participant_id += 1
        self.participant_class[participant_id] = participant_class
        self.status[participant_id] = "arriving"
        return participant_id

    def queue_and_capacity(self, participant_class):
        if participant_class == "A1":
            return self.queue_A1, self.model.k_1
        if participant_class == "A2":
            return self.queue_A2, self.model.k_2
        return self.queue_E, self.model.k_e

    def _observing(self):
        return self.clock >= self.warm_up

    def _cross_warm_up(self, previous_time, next_time):
        if not self._warm_up_done and previous_time < self.warm_up <= next_time:
            self.initial_system_size = (len(self.queue_A1) + len(self.queue_A2)
                                        + len(self.queue_E))
            self._warm_up_done = True

    def _accumulate_queue_area(self, previous_time, next_time):
        
        start = max(previous_time, self.warm_up)
        end = min(next_time, self.termination_time)
        if end > start:
            elapsed = end - start
            self.queue_area["A1"] += elapsed * len(self.queue_A1)
            self.queue_area["A2"] += elapsed * len(self.queue_A2)
            self.queue_area["E"] += elapsed * len(self.queue_E)

    def _update_maximum_queue_lengths(self):
        for name, queue in (("A1", self.queue_A1), ("A2", self.queue_A2), ("E", self.queue_E)):
            if len(queue) > self.maximum_queue_lengths[name]:
                self.maximum_queue_lengths[name] = len(queue)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def results(self):
        def safe(numerator, denominator):
            return numerator / denominator if denominator else 0.0

        matches_1, matches_2 = self.matches["A1"], self.matches["A2"]
        total_matches = matches_1 + matches_2
        duration = self.termination_time - self.warm_up

        total_arrivals = sum(self.arrivals.values())
        total_abandonments = sum(self.abandonments.values())
        total_rejections = sum(self.rejections.values())
        remaining = len(self.queue_A1) + len(self.queue_A2) + len(self.queue_E)

        flow_balance_error = (
            self.initial_system_size + total_arrivals
            - (2 * total_matches + total_abandonments + total_rejections + remaining)
        )

        results = {
            "Policy A": self.policy_A,
            "Total Arrivals": total_arrivals,
            "A1 Arrivals": self.arrivals["A1"],
            "A2 Arrivals": self.arrivals["A2"],
            "Employer Arrivals": self.arrivals["E"],

            # Matching efficiency
            "Throughput": safe(total_matches, duration),
            "Matches": total_matches,
            "A1 Matches": matches_1,
            "A2 Matches": matches_2,
            "A1 Throughput": safe(matches_1, duration),
            "A2 Throughput": safe(matches_2, duration),

            # The matching split: share of all matches taken from class 1.
            "Matching Split p1": safe(matches_1, total_matches),

            "Scans": self.scans,
            "Successful Scan Rate": safe(self.scans_successful, self.scans),
            "Failed Scan Rate": safe(self.scans - self.scans_successful, self.scans),

            "A1 Matching Probability": safe(matches_1, self.arrivals["A1"]),
            "A2 Matching Probability": safe(matches_2, self.arrivals["A2"]),
            "Employer Matching Probability": safe(total_matches, self.arrivals["E"]),

            # Abandonment among admitted participants.
            "Abandonments": total_abandonments,
            "A1 Abandonments": self.abandonments["A1"],
            "A2 Abandonments": self.abandonments["A2"],
            "Employer Abandonments": self.abandonments["E"],
            "A1 Abandonment Rate": safe(self.abandonments["A1"], self.admitted["A1"]),
            "A2 Abandonment Rate": safe(self.abandonments["A2"], self.admitted["A2"]),
            "Employer Abandonment Rate": safe(self.abandonments["E"], self.admitted["E"]),
            "Abandonment Rate": safe(total_abandonments, sum(self.admitted.values())),

            "Rejections": total_rejections,
            "Rejection Rate": safe(total_rejections, total_arrivals),

            "Final Q1": len(self.queue_A1),
            "Final Q2": len(self.queue_A2),
            "Final Qe": len(self.queue_E),
            "Remaining in System": remaining,
            "Initial System Size": self.initial_system_size,
            "Flow Balance Error": flow_balance_error,
            "Max Q1": self.maximum_queue_lengths["A1"],
            "Max Q2": self.maximum_queue_lengths["A2"],
            "Max Qe": self.maximum_queue_lengths["E"],
        }

        labels = {"A1": "A1", "A2": "A2", "E": "Employer"}
        for key, label in labels.items():
            matched_wait = safe(self.wait_matched_total[key], self.wait_matched_count[key])
            departures = self.wait_matched_count[key] + self.wait_abandoned_count[key]
            all_wait = safe(self.wait_matched_total[key] + self.wait_abandoned_total[key],
                            departures)
            queue_length = safe(self.queue_area[key], duration)
            admission_rate = safe(self.admitted[key], duration)

            results[f"Average Wait {label}"] = matched_wait
            results[f"Average Wait All {label}"] = all_wait
            results[f"Average Q{'1' if key == 'A1' else '2' if key == 'A2' else 'e'}"] = queue_length

            # Little's Law
            predicted = admission_rate * all_wait
            results[f"Little L {label}"] = queue_length
            results[f"Little Lambda {label}"] = admission_rate
            results[f"Little W {label}"] = all_wait
            results[f"Little Error {label} (%)"] = (
                100.0 * (queue_length - predicted) / queue_length if queue_length else 0.0
            )

        return results
