"""Reusable analysis components for the A1-1 mission."""

from .pipeline import DataAnalyzer

# 패키지 사용자가 내부 모듈 경로를 몰라도 `from src import DataAnalyzer`로 접근하는
# 최소 공개 API다. 구현 세부 클래스는 의도적으로 노출하지 않는다.
__all__ = ["DataAnalyzer"]
