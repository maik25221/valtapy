from typing import Any, Dict

import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from ...base_privacy import Privacy


class MembershipInferenceAttack(Privacy):
    def __init__(self, data, gen_data, path: str = None):
        super().__init__(data=data, gen_data=gen_data, path=path)
        self.attack_model = None
        self.attack_results = None

    def split_data(self, test_size=0.5):
        """Divide los datos en un conjunto de entrenamiento y un conjunto de prueba."""
        # Convert to numpy arrays
        original_data = self.to_numpy(self.data)
        synthetic_data = self.to_numpy(self.gen_data)

        # Add small amount of noise to avoid perfect duplicates
        # original_data += np.random.normal(size=original_data.shape) * 0.01

        X_train, X_test, y_train, y_test = train_test_split(
            original_data,
            np.ones(len(original_data)),
            test_size=test_size,
            random_state=self.seed - 1,
        )
        X_gen_train, X_gen_test, y_gen_train, y_gen_test = train_test_split(
            synthetic_data,
            np.zeros(len(synthetic_data)),
            test_size=test_size,
            random_state=self.seed + 1,
        )

        X_combined_train = np.concatenate((X_train, X_gen_train))
        y_combined_train = np.concatenate((y_train, y_gen_train))

        X_combined_test = np.concatenate((X_test, X_gen_test))
        y_combined_test = np.concatenate((y_test, y_gen_test))

        return X_combined_train, X_combined_test, y_combined_train, y_combined_test

    def train_attack_model(self, X_train, y_train):
        """Entrena el modelo de ataque."""
        self.attack_model = LogisticRegression()
        self.attack_model.fit(X_train, y_train)

    def evaluate_attack(self, X_test, y_test):
        """Evalúa el modelo de ataque."""
        y_pred = self.attack_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def execute_attack(self):
        """Ejecuta el ataque de membresía."""
        X_train, X_test, y_train, y_test = self.split_data()
        self.train_attack_model(X_train, y_train)
        accuracy = self.evaluate_attack(X_test, y_test)
        self.attack_results = accuracy

    def write_results(self):
        if self.path:
            with open(self.path, "w") as file:
                file.write(
                    f"Membership Inference Attack Accuracy: {self.attack_results}\n"
                )

    def execute(self) -> Dict[str, Any]:
        self.execute_attack()
        if self.path:
            self.write_results()

        return {
            "membership_inference_accuracy": self.attack_results,
            "description": "Membership inference attack success rate - lower is better for privacy",
        }
