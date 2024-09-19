from abc import ABC, abstractmethod, ABCMeta
from modules.ftu import FTU
from typing import List
import logging

class Pipeline(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def run(self):
        """
            this function will act as the core processing of the pipeline
            It will receive list of FTUs
            Will apply the image processing steps needed for Segmentation, 
            Histomorphometry Feature Extraction
        """
        return
    
class NaglahPipeline(Pipeline):
    def __init__(self, config):
        self.name = config["name"]
        self.pipeline = "NaglahPipeline"

    def run(self, ftus: List[FTU]):
        logging.warning(f"Pipeline {self.pipeline} is starting")
        for i in range(len(ftus)):
            ftu = ftus[i]
            logging.warning(f"pipeline name: {self.pipeline} processing {ftu.name}")

class SayatPipeline(Pipeline):
    def __init__(self, config):
        self.name = config["name"]
        self.pipeline = "SayatPipeline"

    def run(self, ftus: List[FTU]):
        logging.warning(f"Pipeline {self.pipeline} is starting")
        for i in range(len(ftus)):
            ftu = ftus[i]
            logging.warning(f"pipeline name: {self.pipeline} processing {ftu.name}")
