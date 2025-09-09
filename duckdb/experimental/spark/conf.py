from typing import Optional, List, Tuple
from duckdb.experimental.spark.exception import ContributionsAcceptedError


class SparkConf:
    def __init__(self) -> None:
        raise NotImplementedError

    def contains(self, key: str) -> bool:
        raise ContributionsAcceptedError

    def get(self, key: str, defaultValue: Optional[str] = None) -> Optional[str]:
        raise ContributionsAcceptedError

    def getAll(self) -> list[tuple[str, str]]:
        raise ContributionsAcceptedError

    def set(self, key: str, value: str) -> "SparkConf":
        raise ContributionsAcceptedError

    def setAll(self, pairs: list[tuple[str, str]]) -> "SparkConf":
        raise ContributionsAcceptedError

    def setAppName(self, value: str) -> "SparkConf":
        raise ContributionsAcceptedError

    def setExecutorEnv(
        self, key: Optional[str] = None, value: Optional[str] = None, pairs: Optional[list[tuple[str, str]]] = None
    ) -> "SparkConf":
        raise ContributionsAcceptedError

    def setIfMissing(self, key: str, value: str) -> "SparkConf":
        raise ContributionsAcceptedError

    def setMaster(self, value: str) -> "SparkConf":
        raise ContributionsAcceptedError

    def setSparkHome(self, value: str) -> "SparkConf":
        raise ContributionsAcceptedError

    def toDebugString(self) -> str:
        raise ContributionsAcceptedError


__all__ = ["SparkConf"]
