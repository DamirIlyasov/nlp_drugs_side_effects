import re


def parseTweetsWithIds(input, output):
    with open(input, "r", encoding='utf-8') as infile, open(output, "w",
                                                            encoding='utf-8') as outfile:
        for line in infile:
            # outfile.write(re.findall("\s((0)|(1))\s.+", line)[0])
            match = re.search(r"((\t)|(\s\s\s))(?P<without_tab>((0)|(1))((\t)|(\s\s\s)).+)", line)
            if match:
                outfile.write(match.group('without_tab') + '\n')


def parseJSONData(input, output):
    with open(input, "r", encoding='utf-8') as infile, open(output, "w",
                                                            encoding='utf-8') as outfile:
        for line in infile:
            match = re.search(r"\"text\":\"(?P<text>.+?)\"", line)
            if match:
                # removing links
                tweetWithoutLink = re.sub(r"(https\S+)|(\\u\S+)", '', match.group('text'))
                outfile.write(tweetWithoutLink + '\n')


# parseTweetsWithIds("loaded_tweets.txt", "loaded_tweets_parsed.txt")


parseJSONData("sources/tweets17.log", "results/json_parsed.txt")
