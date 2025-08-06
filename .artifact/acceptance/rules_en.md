## Goal
Rewrite the `delete_task` test to include a 404 case and add logging to the `login_handler`

## Context
- Testing related to `delete_task`
- Logging enhancements for `login_handler`

## Change Request
**Required:**
- Rewrite the test for `delete_task` to cover 404 scenarios
- Add logging functionality to the `login_handler`

## Constraints
**Unclear requirements (handle carefully):**
- Exact logging details or format for `login_handler`
- Whether the logging should be added in the handler implementation or only in tests

## Acceptance Criteria
- `delete_task` test includes a case for 404 response
- `login_handler` has added logging that can be verified

## Assumptions
- `delete_task` test is located in `test_tasks.py`
- `login_handler` is defined in `test_auth.py` or related authentication module
- Logging refers to adding log statements for monitoring or debugging

## Related (if any)
- test_tasks.py — `delete_task`
- test_auth.py — `login_handler`
