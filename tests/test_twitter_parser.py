from ml.twitter.twitter_parser import TwitterParser


def test_twitter_parser():
    parser = TwitterParser("the quick brown FOX jumped over the Dogs")
    actual_tokens = parser.get_final_token_list()
    expected_tokens = ["quick", "brown", "fox", "jump", "dog"]
    verify_tokens(actual_tokens, expected_tokens)

    parser = TwitterParser(
        "Just got sent this photo from Ruby #Alaska"
        + " as smoke from #wildfires pours into a school"
    )
    actual_tokens = parser.get_final_token_list()
    expected_tokens = [
        "got",
        "sent",
        "photo",
        "rubi",
        "alaska",
        "smoke",
        "wildfir",
        "pour",
        "school",
    ]
    verify_tokens(actual_tokens, expected_tokens)


def verify_tokens(actual_list, expected_list):
    for i in range(0, len(expected_list)):
        expected = expected_list[i]
        actual = actual_list[i]
        assert actual == expected
