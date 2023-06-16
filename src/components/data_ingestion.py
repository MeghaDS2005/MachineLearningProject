import os
import sys
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass                                     ## used to create class variables

@dataclass
class DataIngestionConfig:                                            #artifacts are folders
    train_data_path: str=os.path.join('artifacts',"train.csv")        #these are the inputs to data ingestion components
    test_data_path: str=os.path.join('artifacts',"test.csv")          #data ingestion knoes where to save these three files.
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()                   #as soon as this class is called these 3 paths 
                                                                      #train_data_path, test_data_path  and raw_data_path  will get saved in  self.ingestion_config variable.      
   
   
    def initiate_data_ingestion(self):                                #used if we have to read data from a database like MongoDb and its code will be written in utils
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys) 
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

   # modeltrainer=ModelTrainer()
   # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
   

    