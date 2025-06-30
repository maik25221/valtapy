from typing import Any, Dict

import numpy as np
import pandas as pd

from ...base_privacy import Privacy


class IdentityAttributeDisclosure(Privacy):
    def __init__(self, data, gen_data, path: str = None):
        super().__init__(data=data, gen_data=gen_data, path=path)
        self.attribute_disclosure_rate = None
        self.identity_disclosure_rate = None

    # Eliminar la función to_numpy ya que la heredamos de Privacy

    def simulate_attribute_attack(self, known_attributes_percentage=0.01):
        """
        Simulate an attribute disclosure attack by revealing a small percentage of attributes
        and trying to guess the rest.
        """
        original_data = self.to_numpy(self.data)
        synthetic_data = self.to_numpy(self.gen_data)

        num_attributes = original_data.shape[1]
        num_known_attributes = max(1, int(num_attributes * known_attributes_percentage))

        # Limit samples to avoid long computation
        max_samples = min(500, len(original_data), len(synthetic_data))
        total_comparisons = min(max_samples, len(original_data), len(synthetic_data))

        correct_guesses = 0

        for i in range(total_comparisons):
            try:
                original_record = original_data[i]
                synthetic_record = synthetic_data[i]

                known_indices = np.random.choice(
                    num_attributes, num_known_attributes, replace=False
                )
                known_attributes = original_record[known_indices]
                synthetic_known_attributes = synthetic_record[known_indices]

                # Use allclose for floating point comparison
                if np.allclose(known_attributes, synthetic_known_attributes, rtol=1e-6):
                    correct_guesses += 1
            except (ValueError, TypeError, IndexError):
                # Skip comparison if data types are incompatible or index error
                continue

        self.attribute_disclosure_rate = (
            correct_guesses / total_comparisons if total_comparisons > 0 else 0.0
        )

    def simulate_identity_attack(self):
        """
        Simulate an identity disclosure attack by checking if any synthetic record matches an original record.
        Optimized version using hashing for faster comparison.
        """
        original_data = self.to_numpy(self.data)
        synthetic_data = self.to_numpy(self.gen_data)

        # Limit the comparison to avoid extremely long computation
        max_samples = min(1000, len(original_data), len(synthetic_data))

        # Sample data if datasets are too large
        if len(original_data) > max_samples:
            orig_indices = np.random.choice(
                len(original_data), max_samples, replace=False
            )
            original_sample = original_data[orig_indices]
        else:
            original_sample = original_data

        if len(synthetic_data) > max_samples:
            synth_indices = np.random.choice(
                len(synthetic_data), max_samples, replace=False
            )
            synthetic_sample = synthetic_data[synth_indices]
        else:
            synthetic_sample = synthetic_data

        # Convert to tuples for hashing (much faster than array comparison)
        try:
            # Round to avoid floating point precision issues
            original_tuples = set(tuple(np.round(row, 6)) for row in original_sample)
            synthetic_tuples = set(tuple(np.round(row, 6)) for row in synthetic_sample)

            # Count matches using set intersection
            matches = len(original_tuples.intersection(synthetic_tuples))

        except (TypeError, OverflowError):
            # Fallback for non-numeric data or overflow issues
            matches = 0
            for i, original_record in enumerate(original_sample):
                if i % 100 == 0 and i > 0:  # Progress check every 100 iterations
                    break  # Limit to prevent infinite loops
                for synthetic_record in synthetic_sample:
                    try:
                        if np.allclose(original_record, synthetic_record, rtol=1e-6):
                            matches += 1
                            break
                    except (ValueError, TypeError):
                        continue

        self.identity_disclosure_rate = (
            matches / len(original_sample) if len(original_sample) > 0 else 0.0
        )

    def write_results(self):
        if self.path:
            with open(self.path, "w") as file:
                file.write(
                    f"Attribute Disclosure Rate: {self.attribute_disclosure_rate}\n"
                )
                file.write(
                    f"Identity Disclosure Rate: {self.identity_disclosure_rate}\n"
                )

    def execute(self) -> Dict[str, Any]:
        self.simulate_attribute_attack()
        self.simulate_identity_attack()
        if self.path:
            self.write_results()

        return {
            "attribute_disclosure_rate": self.attribute_disclosure_rate,
            "identity_disclosure_rate": self.identity_disclosure_rate,
            "description": "Identity and attribute disclosure risk assessment",
        }
