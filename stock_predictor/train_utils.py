from __future__ import annotations

from typing import Optional

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model


def train_model(
	model: Model,
	X_train,
	y_train,
	X_val=None,
	y_val=None,
	epochs: int = 25,
	batch_size: int = 32,
	patience: int = 5,
	verbose: int = 1,
):
	"""Train the model with EarlyStopping on validation loss if provided.

	Returns the Keras History object.
	"""
	callbacks = []
	if X_val is not None and y_val is not None:
		callbacks.append(
			EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
		)

	validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
	history = model.fit(
		X_train,
		y_train,
		validation_data=validation_data,
		epochs=epochs,
		batch_size=batch_size,
		verbose=verbose,
		callbacks=callbacks,
	)
	return history
