import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from new_deepimfam.component import DeepImFam

class Split3DeepImFam(DeepImFam):
    def __init__(self, config_path) -> None:
        super().__init__(config_path)
        
    def train(self):
        df = pd.read_csv(self.images_info_path)
        
        # SPLIT DATA
        train_df, test_df = self.split_train_test(df)
        train_df, val_df = self.split_train_test(train_df, test_size=.5)

        # CALL GENERATOR
        image_data_frame_gen = self.ImageDataFrameGenerator(
            images_directory=self.images_directory,
            x_col="path",
            y_col=self.hierarchy_label,
            target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            batch_size=self.BATCH_SIZE
        )
        
        train_gen = image_data_frame_gen.get_generator(df=train_df, shuffle=False)
        test_gen = image_data_frame_gen.get_generator(df=test_df, shuffle=False)
        
        # CALLBACK
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=20,
            min_lr=1e-5
        )

        # モデル
        early_stopping = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=80,
        )

        model = self.generate_model()
        history = model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=1000,
            callbacks=[reduce_lr, early_stopping],
            batch_size=512,
        )    

        # SAVE MODEL
        fname = os.path.join(self.results_directory, "model.h5")
        model.save(fname)

        # SAVE RESULT
        result = history.history
        fname = os.path.join(self.results_directory, "history.csv")
        pd.DataFrame(result).to_csv(fname)
        if not os.path.exists(self.metrics_path):
            metrics = {}
        else:
            with open(self.metrics_path, "r") as f:
                metrics = json.load(f)
        for key in ["loss", "accuracy"]:
            metrics[key] = result[key][-1]
            metrics["val_" + key] = result["val_" + key][-1]
        self.save_obj(metrics, self.metrics_path)
        
    def predict(self):
        df = pd.read_csv(self.images_info_path)
        
        # SPLIT DATA
        train_df, test_df = self.split_train_test(df)
        train_df, val_df = self.split_train_test(train_df)
        
        # SET ImageDataDrameGenerator
        image_data_frame_gen = self.ImageDataFrameGenerator(
            images_directory=self.images_directory,
            x_col="path",
            y_col=self.hierarchy_label,
            target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            batch_size=self.BATCH_SIZE
        )

        train_gen = image_data_frame_gen.get_generator(df=train_df, shuffle=True)
        val_gen = image_data_frame_gen.get_generator(df=val_df, shuffle=True)
        test_gen = image_data_frame_gen.get_generator(df=test_df, shuffle=False)        

        # LOAD MODEL
        model = self.load_model()

        # PREDICT
        train_proba = model.predict(train_gen)
        test_proba = model.predict(test_gen)
        val_proba = model.predict(val_gen)
        train_pred = np.argmax(train_proba, axis=1)
        val_pred = np.argmax(val_proba, axis=1)
        test_pred = np.argmax(test_proba, axis=1)

        # SAVE PREDICT PROBA
        train_fname = os.path.join(self.results, "train_proba_weighted.csv")
        val_fname = os.path.join(self.results, "val_proba_weighted.csv")
        test_fname = os.path.join(self.results, "test_proba_weighted.csv")
        if not os.path.exists(train_fname):
            self.save_dict_as_dataframe({"labels": train_gen.labels}, train_fname)
            self.save_dict_as_dataframe({"labels": val_gen.labels}, val_fname)
            self.save_dict_as_dataframe({"labels": test_gen.labels}, test_fname)

        train_dict = self.load_csv_as_dict(train_fname)
        val_dict = self.load_csv_as_dict(val_fname)
        test_dict = self.load_csv_as_dict(test_fname)
        for i in range(5):
            train_dict["-".join([self.index1, self.index2, str(i)])] = train_proba[:, i]
            val_dict["-".join([self.index1, self.index2, str(i)])] = val_proba[:, i]
            test_dict["-".join([self.index1, self.index2, str(i)])] = test_proba[:, i]
        self.save_dict_as_dataframe(train_dict, train_fname)
        self.save_dict_as_dataframe(val_dict, val_fname)
        self.save_dict_as_dataframe(test_dict, test_fname)

        # SAVE PREDICT LABELS
        train_fname = os.path.join(self.results, "train_predict_weighted.csv")
        val_fname = os.path.join(self.results, "val_predict_weighted.csv")
        test_fname = os.path.join(self.results, "test_predict_weighted.csv")
        if not os.path.exists(train_fname):
            self.save_dict_as_dataframe({"labels": train_gen.labels}, train_fname)
            self.save_dict_as_dataframe({"labels": val_gen.labels}, val_fname)
            self.save_dict_as_dataframe({"labels": test_gen.labels}, test_fname)

        train_dict = self.load_csv_as_dict(train_fname)
        train_dict["-".join([self.index1, self.index2])] = train_pred.tolist()
        self.save_dict_as_dataframe(train_dict, train_fname)
        
        val_dict = self.load_csv_as_dict(val_fname)
        val_dict["-".join([self.index1, self.index2])] = val_pred.tolist()
        self.save_dict_as_dataframe(val_dict, val_fname)

        test_dict = self.load_csv_as_dict(test_fname)
        test_dict["-".join([self.index1, self.index2])] = test_pred.tolist()
        self.save_dict_as_dataframe(test_dict, test_fname)

if __name__ == "__main__":
    config_path = "new_deepimfam/config.yaml"
    train_deepimfam = Split3DeepImFam(config_path=config_path)
    train_deepimfam.train()
    train_deepimfam.predict()
