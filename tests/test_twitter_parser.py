from ml.twitter.twitter_parser import TwitterParser


def test_twitter_parser():
    parser = TwitterParser()
    actual_tokens = parser.normalize("the quick brown foxes "
        + "jumped over the dogs")
    expected_tokens = ["quick", "brown", "fox", "jump", "dog"]
    verify_tokens(actual_tokens, expected_tokens)

    actual_tokens = parser.normalize ("Just got sent this "
        "photo from Ruby #Alaska as smoke from #wildfires pours into a school"
    )
    expected_tokens = [
        "got",
        "send",
        "photo",
        "ruby",
        "alaska",
        "smoke",
        "wildfire",
        "pour",
        "school",
    ]
    verify_tokens(actual_tokens, expected_tokens)


def verify_tokens(actual_list, expected_list):
    for i in range(0, len(expected_list)):
        expected = expected_list[i]
        actual = actual_list[i]
        assert actual == expected
