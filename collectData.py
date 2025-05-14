import tweepy
import json
import time

first_bearer_token = "AAAAAAAAAAAAAAAAAAAAAI4R1AEAAAAA8rlAqqpvXN8qL0uFNoh2EJJ02BM%3DgMZTZyhLkSCjsJBVrr99y5rUUDoO89SEjf3ts6B5i06h30Tfhy" #comes from @test4textm1
second_bearer_token = "AAAAAAAAAAAAAAAAAAAAAKsR1AEAAAAAm5sEx6hXSZur2dxJUw3D4%2FAPPhQ%3DYT7xvErZugTyTEKz5uCmFSuIWABRCMTMZDuyZAE115xrAzk3ls"
third_bearer_token = "AAAAAAAAAAAAAAAAAAAAALwR1AEAAAAAHPVsV%2FWetOa9soK6BDvby3pdXYI%3DOPQtkYN42sZwkafJyjrmvnD83Yd520k3ZLreWEdfqUVX3CuPjx"
bearer_tokens = [
    first_bearer_token,
    second_bearer_token,
    third_bearer_token
]

# Twitter query
query = "(lebron OR jordan) (goat OR best) -is:retweet lang:en"
# Parameters
max_results = 100  # Max per request allowed by Twitter API
total_tweets = 300
collected = 0
token_index = 0
next_token = None

# Initialize Tweepy client
client = tweepy.Client(bearer_token=bearer_tokens[token_index], wait_on_rate_limit=False)

# Output file
output_file = 'goat_debate.json'

with open(output_file, 'w', encoding='utf-8') as f:
    while collected < total_tweets:
        try:
            print(f"🔍 Requesting tweets... Collected: {collected}")
            response = client.search_recent_tweets(
                query=query,
                tweet_fields=['author_id', 'created_at', 'text'],
                max_results=min(max_results, total_tweets - collected),
                next_token=next_token
            )

            # Check if no more tweets found
            if not response.data:
                print("❗ No more tweets found.")
                break

            # Save tweets
            for tweet in response.data:
                tweet_data = {
                    'username': tweet.author_id,
                    'tweet': tweet.text
                }
                json.dump(tweet_data, f, ensure_ascii=False)
                f.write('\n')
                collected += 1

            # Update pagination token
            next_token = response.meta.get('next_token')
            if not next_token:
                print("✅ Finished pagination.")
                break

            time.sleep(5)  # delay to avoid rate limit

        except tweepy.TooManyRequests:
            print("⏳ Rate limit hit. Rotating tokens or waiting...")

            if len(bearer_tokens) > 1:
                # Rotate token
                token_index = (token_index + 1) % len(bearer_tokens)
                client = tweepy.Client(bearer_token=bearer_tokens[token_index], wait_on_rate_limit=False)
                print(f"🔁 Switched to token #{token_index + 1}")
                time.sleep(5)
            else:
                # Wait if only one token
                print("🕒 Waiting 15 minutes due to rate limit...")
                time.sleep(15 * 60)

        except Exception as e:
            print(f"❌ Error occurred: {e}")
            break

print(f"✅ Done! Collected {collected} tweets in total. Saved to '{output_file}'.")