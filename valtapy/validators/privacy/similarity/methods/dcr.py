from typing import Any, Dict

import numpy as np

from valtapy.validators.privacy.base_privacy import Privacy


class DCR(Privacy):
    def __init__(self, data, gen_data, path: str = None):
        super().__init__(data=data, gen_data=gen_data, path=path)
        self.dcr = None

    def calculate_dcr(self) -> float:
        """
        Calculate the Disclosure Control Ratio (DCR)
        """
        # Convert data to numpy arrays for easier manipulation
        original = self.to_numpy(self.data)
        generated = self.to_numpy(self.gen_data)

        # Ensure the arrays are the same length
        if len(original) != len(generated):
            raise ValueError(
                "The length of the original data and generated data must be the same."
            )

        # Calculate the frequency of each unique value in the original data
        unique, counts = np.unique(original, return_counts=True)
        original_freq = dict(zip(unique, counts))

        # Calculate the frequency of each unique value in the generated data
        unique_gen, counts_gen = np.unique(generated, return_counts=True)
        generated_freq = dict(zip(unique_gen, counts_gen))

        # Calculate DCR
        dcr_values = []
        for key in original_freq:
            if key in generated_freq:
                dcr = generated_freq[key] / original_freq[key]
                dcr_values.append(dcr)
            else:
                dcr_values.append(0)

        # The DCR is the mean of all calculated DCR values
        self.dcr = np.mean(dcr_values)
        return self.dcr

    def write_dcr(self):
        if self.path:
            with open(self.path, "w") as file:
                file.write(f"DCR: {self.dcr}\n")

    def execute(self) -> Dict[str, Any]:
        dcr_value = self.calculate_dcr()
        if self.path:
            self.write_dcr()
        return {
            "dcr": dcr_value,
            "description": "Disclosure Control Ratio - measures frequency preservation",
        }
