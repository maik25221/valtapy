from typing import Any, Dict

import numpy as np

from valtapy.validators.privacy.base_privacy import Privacy


class DifferentialPrivacy(Privacy):
    def __init__(self, data, gen_data, path: str = None, epsilon=1.0, delta=1e-5):
        super().__init__(data=data, gen_data=gen_data, path=path)
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_loss = None

    def add_laplace_noise(self, scale: float, size: tuple = (1,)) -> np.ndarray:
        """Add Laplace noise for differential privacy"""
        return np.random.laplace(0, scale, size)

    def calculate_privacy_score(self) -> float:
        """
        Calculate a simple privacy score based on epsilon and delta values.
        Lower epsilon = better privacy (more noise)
        """
        # Simple scoring: normalize epsilon to 0-1 scale where lower is better
        # Typical epsilon values range from 0.1 to 10
        normalized_epsilon = min(self.epsilon / 10.0, 1.0)
        privacy_score = 1.0 - normalized_epsilon

        # Factor in delta (should be very small for good privacy)
        delta_penalty = min(self.delta * 1e5, 0.5)  # Scale delta appropriately
        privacy_score = max(0.0, privacy_score - delta_penalty)

        return privacy_score

    def simulate_differential_privacy(self) -> Dict[str, float]:
        """
        Simulate differential privacy by adding noise to the data and measuring the effect
        """
        original_data = self.to_numpy(self.data)
        synthetic_data = self.to_numpy(self.gen_data)

        # Calculate noise scale based on epsilon
        sensitivity = 1.0  # Assume unit sensitivity for simplicity
        noise_scale = sensitivity / self.epsilon

        # Add noise to original data
        noisy_original = original_data + self.add_laplace_noise(
            noise_scale, original_data.shape
        )

        # Calculate utility loss (how much the data changed)
        mse_original = np.mean((original_data - noisy_original) ** 2)

        # Calculate privacy score
        privacy_score = self.calculate_privacy_score()

        return {
            "privacy_score": privacy_score,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "noise_scale": noise_scale,
            "utility_loss_mse": mse_original,
        }

    def write_results(self, results: Dict[str, float]):
        """Write privacy results to file"""
        if self.path:
            with open(self.path, "w") as file:
                file.write(f"Differential Privacy Results:\n")
                file.write(f"Privacy Score: {results['privacy_score']:.4f}\n")
                file.write(f"Epsilon: {results['epsilon']}\n")
                file.write(f"Delta: {results['delta']}\n")
                file.write(f"Noise Scale: {results['noise_scale']:.4f}\n")
                file.write(f"Utility Loss (MSE): {results['utility_loss_mse']:.4f}\n")

    def execute(self) -> Dict[str, Any]:
        """Execute differential privacy analysis"""
        results = self.simulate_differential_privacy()
        if self.path:
            self.write_results(results)

        return {
            "differential_privacy": results,
            "description": "Differential privacy analysis with Laplace noise mechanism",
        }
