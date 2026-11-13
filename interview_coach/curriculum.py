"""Illustrative role-based interview rounds for the built-in practice experience."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class InterviewRound:
    name: str
    focus: str


INTERVIEW_ROUNDS = (
    InterviewRound(
        "Aptitude & Analytical Reasoning",
        "reasoning, quantitative thinking, and explaining assumptions",
    ),
    InterviewRound(
        "Behavioral & Ownership",
        "specific examples of ownership, collaboration, and learning",
    ),
    InterviewRound(
        "Role Fundamentals",
        "foundational knowledge and role-specific trade-offs",
    ),
    InterviewRound(
        "Hands-on Problem Solving",
        "a practical task, debugging approach, or technical exercise",
    ),
    InterviewRound(
        "Case Study & Design",
        "a realistic case, design choices, validation, and measurable outcomes",
    ),
    InterviewRound(
        "HR, Communication & Culture",
        "motivation, communication, judgement, and working with others",
    ),
)

QUESTIONS_PER_ROUND = 2
SUGGESTED_COMPANIES = (
    "Acme Analytics",
    "Google",
    "Microsoft",
    "Amazon",
    "NVIDIA",
    "Siemens",
    "Northstar Bank",
    "Contoso Cloud",
    "Summit Manufacturing",
)

ROLE_QUESTION_BANK: dict[str, tuple[tuple[str, str], ...]] = {
    "data analyst": (
        (
            "A conversion rate moves from 5% to 6%. How would you calculate and explain the relative change?",
            "A sample has 40 successes in 200 trials. What is the observed rate, and what uncertainty would you communicate?",
        ),
        (
            "Tell me about a time your analysis changed a stakeholder's decision. What evidence made the difference?",
            "Describe a time you found a data-quality issue late in an analysis. How did you handle it?",
        ),
        (
            "How would you define an activation rate so that teams calculate it consistently?",
            "When would you use a median instead of a mean to describe customer behavior?",
        ),
        (
            "Write a SQL approach to find the top three products by revenue in each region.",
            "A dashboard metric is unexpectedly null for one segment. How would you debug the data pipeline?",
        ),
        (
            "Sign-ups increased after a launch but activation fell. How would you investigate and recommend a next step?",
            "How would you design a measurement plan for a new feature with low early adoption?",
        ),
        (
            "How do you explain a result when the data is incomplete or inconclusive?",
            "What would you do if a leader requested a metric that you believe could mislead the team?",
        ),
    ),
    "software engineer": (
        (
            "How would you estimate the memory growth of a list that stores one million records?",
            "A service handles 120 requests per second and traffic doubles. What would you measure first?",
        ),
        (
            "Tell me about a production issue you owned. How did you restore service and prevent recurrence?",
            "Describe a time you changed your technical approach after receiving code-review feedback.",
        ),
        (
            "What does idempotency mean for an API, and when is it important?",
            "How would you choose between a relational database and a key-value store for a feature?",
        ),
        (
            "How would you detect and safely handle duplicate events in a message consumer?",
            "A deployment causes elevated latency but no errors. How would you investigate it?",
        ),
        (
            "Design a notification service that can retry delivery without sending duplicates.",
            "How would you scale a read-heavy service while keeping its data reasonably fresh?",
        ),
        (
            "How do you communicate a delivery risk when a project deadline is approaching?",
            "Tell me about a time you disagreed with a design decision. How was it resolved?",
        ),
    ),
    "ai ml engineer": (
        (
            "A classifier has 99% accuracy on a dataset where 99% of examples are negative. What does that tell you?",
            "If a model's precision rises while recall falls, what practical trade-off might that represent?",
        ),
        (
            "Tell me about a model or experiment that did not work as expected. What did you learn?",
            "Describe how you aligned model evaluation with the needs of a product or research partner.",
        ),
        (
            "What is data leakage, and how can it lead to misleading offline evaluation?",
            "How would you choose a metric for a ranking or recommendation model?",
        ),
        (
            "A model's input feature distribution shifts after deployment. How would you diagnose the issue?",
            "Outline a reproducible training pipeline from raw data through model evaluation.",
        ),
        (
            "Design an inference service for a recommendation model with a strict latency budget.",
            "How would you decide whether a model improvement is meaningful enough to launch?",
        ),
        (
            "How do you explain model uncertainty and limitations to a non-technical stakeholder?",
            "What checks would you add before using a model in a high-impact decision?",
        ),
    ),
    "devops engineer": (
        (
            "A service has 99.9% monthly availability. Approximately how much downtime does that allow in a 30-day month?",
            "A deployment queue is growing while the number of workers stays fixed. What capacity signals would you inspect?",
        ),
        (
            "Tell me about an incident you helped resolve. How did you coordinate recovery and follow-up?",
            "Describe a time you improved a repetitive operational task through automation.",
        ),
        (
            "What is the difference between a container image and a running container?",
            "How do infrastructure as code and configuration management solve different problems?",
        ),
        (
            "A new release repeatedly restarts in production. What is your safe, step-by-step triage plan?",
            "A CI pipeline passes locally but fails in the build environment. What would you compare?",
        ),
        (
            "Design a deployment process with gradual rollout, health checks, and fast rollback.",
            "How would you improve reliability for a service with a recurring dependency outage?",
        ),
        (
            "How do you balance on-call reliability work with feature delivery?",
            "What information should an incident update include for non-technical stakeholders?",
        ),
    ),
    "mechanical engineer": (
        (
            "A shaft turns at 120 revolutions per minute. How many revolutions does it complete in 30 seconds?",
            "A part is measured at 10.2 mm with a tolerance of plus or minus 0.1 mm. Is a 10.31 mm part acceptable?",
        ),
        (
            "Tell me about a design or build issue you helped resolve. What evidence guided your decision?",
            "Describe a time you raised a safety or manufacturability concern during a project.",
        ),
        (
            "How would you choose between aluminum and steel for a lightweight structural component?",
            "What factors determine an appropriate fit and tolerance between two mating parts?",
        ),
        (
            "A rotating assembly develops unexpected vibration after maintenance. How would you isolate the cause?",
            "How would you investigate a component that is overheating only under peak load?",
        ),
        (
            "A bracket must become lighter without exceeding its deflection limit. How would you approach redesign?",
            "How would you validate a prototype before releasing a revised part to production?",
        ),
        (
            "How do you communicate a safety-critical design limitation to a project manager?",
            "What would you do if a production shortcut conflicted with an approved engineering specification?",
        ),
    ),
    "product manager": (
        (
            "A feature has 1,000 users and 200 weekly active users. What additional context is needed before judging adoption?",
            "If a funnel converts 20% of visitors and traffic doubles, what would you avoid assuming about outcomes?",
        ),
        (
            "Tell me about a time you aligned stakeholders around a difficult product trade-off.",
            "Describe a product decision you made with incomplete information. How did you reduce the risk?",
        ),
        (
            "How would you prioritize two roadmap requests with similar reach but different strategic value?",
            "What is the difference between an output metric and an outcome metric?",
        ),
        (
            "How would you test whether a change to onboarding improves activation?",
            "A feature request is popular with one customer but conflicts with the broader roadmap. What would you do?",
        ),
        (
            "Sign-ups rise after a launch but retention falls. How would you diagnose the result and decide what to do?",
            "How would you define success and guardrails for a redesigned checkout experience?",
        ),
        (
            "How do you keep customers informed when a requested feature will not be built?",
            "What would you want to learn about a team before deciding to join it?",
        ),
    ),
}

SUGGESTED_ROLES = (
    "Data Analyst",
    "Software Engineer",
    "AI/ML Engineer",
    "DevOps Engineer",
    "Mechanical Engineer",
    "Product Manager",
)


def _normalize_role(role: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", role.casefold()))


def build_role_round_csv(company: str, role: str) -> bytes | None:
    questions = ROLE_QUESTION_BANK.get(_normalize_role(role))
    if questions is None:
        return None

    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("company", "role", "type", "content", "source", "round"))
    for interview_round, round_questions in zip(INTERVIEW_ROUNDS, questions):
        for question in round_questions:
            writer.writerow(
                (
                    company.strip(),
                    role.strip(),
                    "question",
                    question,
                    "Illustrative role-round practice pack",
                    interview_round.name,
                )
            )
    return output.getvalue().encode("utf-8")
