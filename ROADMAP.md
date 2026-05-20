# Job Scout Roadmap

## Positioning

Job Scout should become:

`A resume-grounded job copilot that finds jobs, explains why they matter, and helps the user improve their odds of getting interviews.`

That is a stronger product identity than a generic "AI job search agent."

## V1 Quick Wins

- Completed: Refactor the codebase into clear modules:
  - `resume parsing`
  - `job search`
  - `ranking/scoring`
  - `session state`
- Completed: Add typed models for:
  - `JobPosting`
  - `ResumeProfile`
  - `JobMatchScore`
  - `SearchContext`
- Completed: Return structured scoring evidence:
  - matched resume evidence
  - missing hard requirements
  - preferred vs required gaps
- Completed: Add deterministic evaluation fixtures for:
  - known good matches
  - bad matches
  - entry-level filtering
- Completed: Improve result summaries so each job includes:
  - fit verdict
  - reason to apply
  - blocker risk

## V2 Product Differentiators

- Tailor the resume per job:
  - rewrite summary
  - reorder skills
  - suggest bullet updates
- Generate application assets:
  - cover letter draft
  - recruiter outreach message
  - interview intro pitch
- Add skill-gap guidance:
  - apply now / apply later / skip
  - 7-day upskilling plan
  - 30-day upskilling plan
- Add user preference memory:
  - saved companies
  - liked roles
  - disliked locations
  - preferred work style

## V3 Moat Features

- Multi-source job aggregation:
  - Indeed
  - LinkedIn
  - Wellfound
  - Remote-only boards
  - direct company career pages
- Portfolio-aware scoring:
  - GitHub projects
  - deployed apps
  - LinkedIn/about text
  - case studies
- Application workflow tracking:
  - saved
  - applied
  - interview
  - rejected
  - follow-up reminders
- Company intelligence enrichment:
  - startup vs enterprise
  - likely tech stack
  - remote friendliness
  - hiring urgency
  - fit confidence

## Recommended Next Build Steps

1. Finish modularizing `job_scout/tools.py` into smaller internal modules.
2. Add typed domain models and stop passing raw dicts everywhere.
3. Upgrade scoring output to include evidence and blocker detection.
4. Add resume-tailoring tools so the agent helps with applying, not only searching.
5. Build an evaluation harness so ranking quality can be measured over time.
