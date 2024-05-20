from typing import Union
from abc import ABC, abstractmethod


class BaseTester(ABC):
    @abstractmethod
    def run_test(self) -> Union[dict, list[dict]]:
        pass
