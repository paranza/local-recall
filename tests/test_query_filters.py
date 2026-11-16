from datetime import date

import pytest

from local_recall.query_filters import parse_query

TODAY = date(2001, 5, 16)  # a Wednesday


@pytest.mark.parametrize(
    "query,sender,date_from,date_to,text",
    [
        ("emails from greg whalley in June 2000 about the QQQ short",
         "greg.whalley", date(2000, 6, 1), date(2000, 6, 30), "emails about the QQQ short"),
        ("from tim.belden@enron.com before 2001-03-01 forecast",
         "tim.belden@enron.com", None, date(2001, 2, 28), "forecast"),
        ("underscheduling penalty after March 2000", None, date(2000, 4, 1), None, "underscheduling penalty"),
        ("budget since 2000-12-13", None, date(2000, 12, 13), None, "budget"),
        ("meeting notes in 2001", None, date(2001, 1, 1), date(2001, 12, 31), "meeting notes"),
        ("contract between Jan 2000 and Mar 2000", None, date(2000, 1, 1), date(2000, 3, 31), "contract"),
        ("what did Sara send yesterday", None, date(2001, 5, 15), date(2001, 5, 15), "what did Sara send"),
        ("gas prices last week", None, date(2001, 5, 7), date(2001, 5, 13), "gas prices"),
        ("invoices last month", None, date(2001, 4, 1), date(2001, 4, 30), "invoices"),
        ("video conference on 2000-06-30", None, date(2000, 6, 30), date(2000, 6, 30), "video conference"),
        ("sent by Karen Denne about the press release", "karen.denne", None, None, "about the press release"),
    ],
)
def test_parse_query(query, sender, date_from, date_to, text):
    p = parse_query(query, today=TODAY)
    assert (p.sender, p.date_from, p.date_to, p.text) == (sender, date_from, date_to, text)


@pytest.mark.parametrize(
    "query",
    [
        "When is the video conference with Avi from Mars Corp. scheduled?",  # "from" here is not a sender filter we can trust
        "What are the proposed changes to the underscheduling penalty?",
        "flight in the morning",
        "report on 2000-13-45",  # not a real date
    ],
)
def test_no_date_filter_is_invented(query):
    p = parse_query(query, today=TODAY)
    assert p.date_from is None and p.date_to is None and p.sender is None
