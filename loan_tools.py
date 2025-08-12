# tools/loan_tools.py
from __future__ import annotations
from langchain_core.tools import tool
from typing import Dict, List

def _round2(x: float) -> float:
    return round(float(x) + 1e-12, 2)  # 이진부동소수 오차 완화용 미세 보정

@tool("equal_principal_schedule")
def equal_principal_schedule(
    principal: float,
    annual_rate: float,
    months: int,
    *,
    round_to_cents: bool = True,
) -> Dict:
    """
    원금균등분할상환 스케줄을 계산합니다.

    Args:
        principal: 대출원금(원화 등 통화단위)
        annual_rate: 연이자율(%, 예: 6.5 → 6.5)
        months: 상환기간(개월)
        round_to_cents: 회차별 금액을 소수 둘째 자리(분/센트)로 반올림할지 여부. 기본 True.

    Returns:
        {
          "schedule": [
             {"month": 1, "principal_payment": ..., "interest_payment": ..., "total_payment": ..., "remaining_principal": ...},
             ...
          ],
          "totals": {"total_interest": ..., "total_payment": ...},
          "meta": {"scheme": "EQUAL_PRINCIPAL", "monthly_rate": ..., "principal": ..., "months": ...}
        }

    주의:
        - 매월 원금상환액은 동일하며(= principal / months), 잔액이 줄수록 이자액이 감소합니다.
        - 마지막 회차는 반올림 누적 오차를 보정하여 잔액이 정확히 0이 되도록 조정합니다.
    """
    if months <= 0:
        raise ValueError("months는 1 이상이어야 합니다.")
    if principal < 0:
        raise ValueError("principal은 0 이상이어야 합니다.")
    r = (annual_rate / 100.0) / 12.0

    base_principal = principal / months
    schedule: List[Dict] = []
    remaining = float(principal)
    total_interest = 0.0
    total_payment = 0.0

    for m in range(1, months + 1):
        principal_pay = base_principal if m < months else remaining  # 마지막 회차에 보정
        interest_pay = remaining * r
        payment = principal_pay + interest_pay
        next_remaining = remaining - principal_pay

        if round_to_cents:
            interest_pay = _round2(interest_pay)
            principal_pay = _round2(principal_pay)
            payment = _round2(principal_pay + interest_pay)
            next_remaining = _round2(remaining - principal_pay)

            # 마지막 회차 잔액 보정(±1원/1센트 수준)
            if m == months and abs(next_remaining) >= 0.01:
                principal_pay = _round2(principal_pay + next_remaining)
                payment = _round2(principal_pay + interest_pay)
                next_remaining = 0.0

        schedule.append(
            {
                "month": m,
                "principal_payment": principal_pay,
                "interest_payment": interest_pay,
                "total_payment": payment,
                "remaining_principal": max(0.0, next_remaining),
            }
        )
        remaining = next_remaining
        total_interest += interest_pay
        total_payment += payment

    return {
        "schedule": schedule,
        "totals": {
            "total_interest": _round2(total_interest) if round_to_cents else total_interest,
            "total_payment": _round2(total_payment) if round_to_cents else total_payment,
        },
        "meta": {
            "scheme": "EQUAL_PRINCIPAL",
            "monthly_rate": r,
            "principal": principal,
            "months": months,
        },
    }

@tool("equal_payment_schedule")
def equal_payment_schedule(
    principal: float,
    annual_rate: float,
    months: int,
    *,
    round_to_cents: bool = True,
) -> Dict:
    """
    원리금균등분할상환 스케줄을 계산합니다.

    Args:
        principal: 대출원금(원화 등 통화단위)
        annual_rate: 연이자율(%, 예: 6.5 → 6.5)
        months: 상환기간(개월)
        round_to_cents: 회차별 금액을 소수 둘째 자리(분/센트)로 반올림할지 여부. 기본 True.

    Returns:
        {
          "schedule": [
             {"month": 1, "principal_payment": ..., "interest_payment": ..., "total_payment": ..., "remaining_principal": ...},
             ...
          ],
          "totals": {"total_interest": ..., "total_payment": ...},
          "meta": {"scheme": "EQUAL_PAYMENT", "monthly_rate": ..., "fixed_payment": ..., "principal": ..., "months": ...}
        }

    특징:
        - 매월 상환총액(원리금)이 동일합니다.
        - 초기에는 이자 비중이 높고, 갈수록 원금 비중이 커집니다.
        - 마지막 회차는 반올림 누적 오차를 보정하여 잔액 0이 되도록 조정합니다.
    """
    if months <= 0:
        raise ValueError("months는 1 이상이어야 합니다.")
    if principal < 0:
        raise ValueError("principal은 0 이상이어야 합니다.")

    r = (annual_rate / 100.0) / 12.0
    if r == 0:
        fixed_payment = principal / months
    else:
        # 고정 상환액 A = P * r * (1+r)^n / ((1+r)^n - 1)
        factor = (1 + r) ** months
        fixed_payment = principal * r * factor / (factor - 1)

    if round_to_cents:
        fixed_payment = _round2(fixed_payment)

    schedule: List[Dict] = []
    remaining = float(principal)
    total_interest = 0.0
    total_payment = 0.0

    for m in range(1, months + 1):
        interest_pay = remaining * r
        principal_pay = fixed_payment - interest_pay if r > 0 else fixed_payment
        next_remaining = remaining - principal_pay

        if round_to_cents:
            interest_pay = _round2(interest_pay)
            principal_pay = _round2(fixed_payment - interest_pay) if r > 0 else _round2(principal_pay)
            payment = _round2(principal_pay + interest_pay)
            next_remaining = _round2(remaining - principal_pay)

            # 마지막 회차 잔액 보정
            if m == months and abs(next_remaining) >= 0.01:
                principal_pay = _round2(principal_pay + next_remaining)
                payment = _round2(principal_pay + interest_pay)
                next_remaining = 0.0
        else:
            payment = principal_pay + interest_pay

        schedule.append(
            {
                "month": m,
                "principal_payment": principal_pay,
                "interest_payment": interest_pay,
                "total_payment": payment,
                "remaining_principal": max(0.0, next_remaining),
            }
        )

        remaining = next_remaining
        total_interest += interest_pay
        total_payment += payment

    return {
        "schedule": schedule,
        "totals": {
            "total_interest": _round2(total_interest) if round_to_cents else total_interest,
            "total_payment": _round2(total_payment) if round_to_cents else total_payment,
        },
        "meta": {
            "scheme": "EQUAL_PAYMENT",
            "monthly_rate": r,
            "fixed_payment": fixed_payment,
            "principal": principal,
            "months": months,
        },
    }


@tool("bullet_repayment_schedule")
def bullet_repayment_schedule(
    principal: float,
    annual_rate: float,
    months: int,
    *,
    round_to_cents: bool = True,
) -> Dict:
    """
    만기일시상환(거치상환) 스케줄을 계산합니다.
    - 매월 이자만 납부, 마지막 회차에 원금 전액 상환.
    - 마지막 회차에서 반올림 누적 오차 보정.
    """
    if months <= 0:
        raise ValueError("months는 1 이상이어야 합니다.")
    if principal < 0:
        raise ValueError("principal은 0 이상이어야 합니다.")
    r = (annual_rate / 100.0) / 12.0

    schedule: List[Dict] = []
    remaining = float(principal)
    total_interest = 0.0
    total_payment = 0.0

    for m in range(1, months + 1):
        interest_pay = remaining * r
        principal_pay = 0.0 if m < months else remaining  # 마지막 달에 원금 전액
        payment = principal_pay + interest_pay
        next_remaining = remaining - principal_pay  # 마지막 달에만 0으로

        if round_to_cents:
            interest_pay = _round2(interest_pay)
            principal_pay = _round2(principal_pay)
            payment = _round2(principal_pay + interest_pay)
            next_remaining = _round2(next_remaining)
            # 혹시 잔액이 ±0.01 남으면 마지막 달에 보정
            if m == months and abs(next_remaining) >= 0.01:
                principal_pay = _round2(principal_pay + next_remaining)
                payment = _round2(principal_pay + interest_pay)
                next_remaining = 0.0

        schedule.append(
            {
                "month": m,
                "principal_payment": principal_pay,
                "interest_payment": interest_pay,
                "total_payment": payment,
                "remaining_principal": max(0.0, next_remaining),
            }
        )
        remaining = next_remaining
        total_interest += interest_pay
        total_payment += payment

    return {
        "schedule": schedule,
        "totals": {
            "total_interest": _round2(total_interest) if round_to_cents else total_interest,
            "total_payment": _round2(total_payment) if round_to_cents else total_payment,
        },
        "meta": {
            "scheme": "BULLET",
            "monthly_rate": r,
            "principal": principal,
            "months": months,
        },
    }


if __name__ == "__main__":
    P = 100_000_000
    RATE = 6.0
    N = 12

    print("\n=== 원금균등분할상환 테스트 ===")
    res_principal = equal_principal_schedule.invoke(
        {"principal": P, "annual_rate": RATE, "months": N}
    )
    print(f"총이자: {res_principal['totals']['total_interest']}, 총상환액: {res_principal['totals']['total_payment']}")
    for row in res_principal["schedule"][:3]:
        print(row)
    print("... (중략) ...")
    print(res_principal["schedule"][-1])

    print("\n=== 원리금균등분할상환 테스트 ===")
    res_equal = equal_payment_schedule.invoke(
        {"principal": P, "annual_rate": RATE, "months": N}
    )
    print(f"총이자: {res_equal['totals']['total_interest']}, 총상환액: {res_equal['totals']['total_payment']}")
    for row in res_equal["schedule"][:3]:
        print(row)
    print("... (중략) ...")
    print(res_equal["schedule"][-1])

    print("\n=== 만기일시상환 테스트 ===")
    res_bullet = bullet_repayment_schedule.invoke(
        {"principal": P, "annual_rate": RATE, "months": N}
    )
    print(f"총이자: {res_bullet['totals']['total_interest']}, 총상환액: {res_bullet['totals']['total_payment']}")
    for row in res_bullet["schedule"][:3]:
        print(row)
    print("... (중략) ...")
    print(res_bullet["schedule"][-1])