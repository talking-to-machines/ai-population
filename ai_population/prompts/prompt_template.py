# Tiktok Profile and Tweet Prompt Templates
tiktok_video_prompt_template = """Creation Date: {video_creation_date}
Video Description: {video_description}
Video Duration: {video_duration}
Number of Likes: {num_likes}
Number of Shares: {num_shares}
View Count: {view_count}
Number of Saves: {num_saves}
Number of Comments: {num_comments}
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): {total_engagement_over_num_views}
Tagged Users: {tagged_users}
Hashtags: {hashtags}
Video Transcript: {video_transcript}"""

tiktok_profile_prompt_template = """- Profile Image: {profile_image}
- Profile Name: {profile_name}
- Profile Nickname: {profile_nickname}
- Profile Biography: {profile_biography}
- Profile Signature: {profile_signature}
- Profile Biography Link: {profile_bio_link}
- Profile URL: {profile_url}
- Profile Language: {profile_lang}
- Profile Creation Date: {profile_creation}
- Verified Status: {verified_status}
- Number of Followers: {num_followers} Followers
- Following: {num_following} Users
- Total Number of Likes: {num_likes}
- Total Number of Videos: {num_videos}
- Total Number of Digg: {num_digg}
- Private Account: {private_account}
- Region: {region}
- TikTok Seller: {tiktok_seller}
- Average Engagement Rate: {awg_engagement_rate}
- Comment Engagement Rate: {comment_engagement_rate}
- Like Engagement Rate: {like_engagement_rate}
- Video Transcripts (Sorted from Newest to Oldest):
{video_transcripts}"""


# X (formerly Twitter) Profile and Tweet Prompt Templates
x_tweet_prompt_template = """Creation Date: {created_at} 
Tweet Text: {text}
Number of Likes: {like_count}
Number of Views: {view_count}
Number of Retweets: {retweet_count}
Number of Replies: {reply_count}
Number of Quotes: {quote_count}
Number of Bookmarks: {bookmark_count}
Language: {lang}
Tagged Users: {tagged_users}
Hashtags: {hashtags}"""

x_profile_prompt_template = """- Profile Image: {profile_picture}
- Profile Name: {name}
- Profile ID: {account_id}
- Location: {location}
- Profile Description: {description}
- Profile External Link: {url}
- Profile Creation Date: {created_at}
- Verified Profile: {is_verified}
- Blue Verified Profile: {is_blue_verified}
- Protected Profile: {protected}
- Number of Followers: {followers} Followers
- Following: {following} Users
- Total Number of Tweets: {statuses_count}
- Number of Favorites: {favourites_count}
- Number of Media Content: {media_count}
- Tweets (Sorted from Newest to Oldest):
{tweets}"""


# Market Signals Expert Reflection Prompts
base_expert_reflection_system_prompt = """You are an expert {expert_role} (with a PhD) analyzing the {platform} profile of an influencer with the following details:
{profile_prompt_template}
"""

base_expert_reflection_user_prompt = """Drawing on your expertise as an expert {expert_role} (with a PhD), list out a set of 5-10 observations about this profile. Each observation should help answer or provide additional insights into the following questions:
- Indicate on a scale of 0 to 100, how influential this influencer is (0 means not at all influential and 100 means very influential with millions of followers and mainstream recognition)?
- Indicate on a scale of 0 to 100, how credible or authoritative this influencer is (0 means not at all credible or authoritative and 100 means very credible and authoritative)?
- Which of these areas of finance are the primary focus of the influencer’s posts?
- Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's individual stock predictions (0 means very low quality and 100 means very high quality)?
- Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's evaluation of market sentiment (0 means very low quality and 100 means very high quality)?
- Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's broader evaluation of the economy (0 means very low quality and 100 means very high quality)?
- Who is the influencer’s target audience?

Guidelines
1. Base every observation only on the supplied profile data (follower counts, engagement metrics, bio, post excerpts, transcripts).  
2. Cite concrete evidence in each observation as much as possible 
3. Cover a range of angles based on your area of expertise: reach, credibility markers, content themes, analytical style, time horizon, sector bias, red-flag behaviour, etc.  
4. Use concise, professional language; one or two sentences per observation.  
5. Do not score the questions—just provide observations rich enough for someone else to score them."""

tiktok_portfoliomanager_reflection_system_prompt = (
    base_expert_reflection_system_prompt.format(
        expert_role="Portfolio Manager",
        platform="TikTok",
        profile_prompt_template=tiktok_profile_prompt_template,
    )
)

x_portfoliomanager_reflection_system_prompt = (
    base_expert_reflection_system_prompt.format(
        expert_role="Portfolio Manager",
        platform="X (formerly Twitter)",
        profile_prompt_template=x_profile_prompt_template,
    )
)

portfoliomanager_reflection_user_prompt = base_expert_reflection_user_prompt.format(
    expert_role="Portfolio Manager"
)

tiktok_investmentadvisor_reflection_system_prompt = (
    base_expert_reflection_system_prompt.format(
        expert_role="Investment Advisor",
        platform="TikTok",
        profile_prompt_template=tiktok_profile_prompt_template,
    )
)

x_investmentadvisor_reflection_system_prompt = (
    base_expert_reflection_system_prompt.format(
        expert_role="Investment Advisor",
        platform="X (formerly Twitter)",
        profile_prompt_template=x_profile_prompt_template,
    )
)

investmentadvisor_reflection_user_prompt = base_expert_reflection_user_prompt.format(
    expert_role="Investment Advisor"
)

tiktok_financialanalyst_reflection_system_prompt = (
    base_expert_reflection_system_prompt.format(
        expert_role="Chartered Financial Analyst",
        platform="TikTok",
        profile_prompt_template=tiktok_profile_prompt_template,
    )
)

x_financialanalyst_reflection_system_prompt = (
    base_expert_reflection_system_prompt.format(
        expert_role="Chartered Financial Analyst",
        platform="X (formerly Twitter)",
        profile_prompt_template=x_profile_prompt_template,
    )
)

financialanalyst_reflection_user_prompt = base_expert_reflection_user_prompt.format(
    expert_role="Chartered Financial Analyst"
)

tiktok_economist_reflection_system_prompt = base_expert_reflection_system_prompt.format(
    expert_role="Economist",
    platform="TikTok",
    profile_prompt_template=tiktok_profile_prompt_template,
)

x_economist_reflection_system_prompt = base_expert_reflection_system_prompt.format(
    expert_role="Economist",
    platform="X (formerly Twitter)",
    profile_prompt_template=x_profile_prompt_template,
)

economist_reflection_user_prompt = base_expert_reflection_user_prompt.format(
    expert_role="Economist"
)


# Market Signals Onboarding Prompt Templates
base_finfluencer_onboarding_system_prompt = """You are analyzing a social media profile on {platform} to identify individuals who may be financial influencers (finfluencer).

DEFINITION OF A FINFLUENCER
A finfluencer is a creator whose content includes providing:
1) Macroeconomic analysis on the financial markets (e.g., interest-rate outlook, inflation prints, jobs data, earnings season)
2) Business-related news and current events that have an impact on the US and global economy (e.g., Federal Reserve policy, rate cutes, market-moving news)
3) Opinions, predictions, or recommendations about different financial assets:
    • Individual equities/stocks (e.g., stock market, S&P 500, tickers, options, day-trading setups, etc.)
    • Crypto-assets (e.g., Bitcoin, Ethereum, alt-coins)
    • Foreign-exchange (e.g., FX / forex currency pairs)

HALLMARK SIGNALS OF A FINFLUENCER
• Typically say or imply “buy/sell/hold” or “don’t miss this trade,” share price targets, or promote certain stocks, crypto-assets, FX currencies.
• Frequent use of market-specific terms: “stock picks,” “entry level,” “S&P 500,” “BTC,” “NFP report,” “ticker-symbol ____,” “options flow,” “FX pair,” “rate-cut,” “Federal Reserve,” etc.  
• Explicit or implicit recommendations/predictions: “I’m buying,” “price target $150,” “this coin will 10×,” “short before earnings,” “don’t miss this swing.”  
• News organizations or journalists covering business news and current events that have an impact on US or global financial system.

DEFINITION OF A NON-FINFLUENCER
A non-finfluencer is any profile that:
• Only focuses on stock market education and teaching trading mechanics for different financial assets (e.g., “how to place a limit order,” “what is leverage”, trading strategies, candlestick patterns, technical analysis, chart patterns, risk management, stop-loss techniques) **but do not issue any actionable calls, price views, or stock recommendations**.
• Focuses on personal-finance hygiene (budgeting, saving, couponing, credit-score repair, basic money habits) without market picks
• Discusses topics completely unrelated to finance and the economy

HALLMARK SIGNS OF A NON-FINFLUENCER
• Content limited to household budgeting, debt payoff tips, generic ‘save money’ hacks, student-loan talk, couponing, side-hustles.  
• Purely academic or definitional explainers without advice (“what is a candlestick,” “history of Bitcoin,” “how the Fed works”).  
• “Not financial advice” disclaimers with no actual recommendation content.  
• Focus on trading psychology, motivation, or platform tutorials **only**, with no picks or directional calls.
• Promotional or affiliate language: coupon codes for brokerages, screenshots of profits, paid Discord, Patreon, master-class, free trading courses, workshops.  
• Urgency or emotional triggers: “last chance,” “100 % ROI,” “financial freedom,” “passive income NOW.”  
• Motivational posts or visual cues of wealth purportedly earned from markets (luxury cars, profit dashboards, bold green/red P&L screenshots).

Below are three {platform} profiles and their recent posts that exemplify a financial influencer:
{finfluencer_examples}

Below are three {platform} profiles and their recent posts that exemplify a non-financial influencer:
{nonfinfluencer_examples}

You are also provided high-level and abstract “expert reflections” from a professional portfolio manager, an investment advisor, a chartered financial analyst, and an economist regarding the profile you are analyzing and its content. These reflections are provided below:
{expert_reflection_prompt_template}

Here are the details of the {platform} profile you will be analyzing:
{profile_prompt_template}

Instructions
Analyze the provided information and answer the following questions based strictly on the available data and definitions provided to you. Do not infer or assume any details beyond what is given. Keep responses concise, precise and data-driven."""

tiktok_finfluencer_examples = """Example Finfluencer Profile 1:
- Profile Image: https://p16-sign-va.tiktokcdn.com/tos-maliva-avt-0068/a089791b375beffd40eaf995a4541032~tplv-tiktokx-cropcenter:720:720.jpeg?dr=14579&refresh_token=3963c468&x-expires=1748991600&x-signature=4ZYa6NLuLm7QPpnYYhJOnEEeeUo%3D&t=4d5b0474&ps=13740610&shp=a5d48078&shcp=81f88b70&idc=my
- Profile Name: fung.money
- Profile Nickname: ⚫️ Derrick | Money Professor😎
- Profile Biography: Ex-Wall St 🇺🇸🇨🇦
Bloomberg/CNBC🛸
Tips to Grow Your💰
Portfolio & Signals👇🏻
- Profile Signature: Ex-Wall St 🇺🇸🇨🇦
Bloomberg/CNBC🛸
Tips to Grow Your💰
Portfolio & Signals👇🏻
- Profile Biography Link: nan
- Profile URL: https://www.tiktok.com/@fung.money
- Profile Language: en
- Profile Creation Date: 2022-12-10T20:23:41.000Z
- Verified Status: False
- Number of Followers: 231800.0 Followers
- Following: 183.0 Users
- Total Number of Likes: 1500000.0
- Total Number of Videos: 171.0
- Total Number of Digg: 0.0
- Private Account: False
- Region: US
- TikTok Seller: False
- Average Engagement Rate: 0.044951153664171
- Comment Engagement Rate: 0.0015309360126916
- Like Engagement Rate: 0.0434202176514793
- Video Transcripts (Sorted from Newest to Oldest):
Creation Date: 2025-05-29 22:41:15+00:00
Video Description: 05/29 - Target’s sales are falling and it’s not just the boycotts: In this video, I break down the real reasons behind the decline — including rising competition from Dollar General, Amazon’s same-day delivery, and TikTok Shop stealing impulse purchases. #target #amazon #retail #investing #money #trends #stocks #financialfreedom #creatorsearchinsights 
Video Duration: 76.0
Number of Likes: 2971.0
Number of Shares: 134.0
View Count: 40300.0
Number of Saves: 264.0
Number of Comments: 199.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.08853598014888338
Tagged Users: 
Hashtags: target, amazon, retail, investing, money, trends, stocks, financialfreedom, creatorsearchinsights
Video Transcript: Things are not looking good for Target based on real-time data I have access to. They had a pretty bad earnings miss last week. They blamed the boycotts and you can see here that it hasn't stopped. In fact, it looks like sales are starting to really drop off a cliff. C-score is minus 78, meaning we are quite bearish on the stock based on this data. I dug into it a bit more because that spike down is pretty massive and it looks like there's three other things happening that's not helping the company. First is Dollar General is taking market share. A lot of consumers wanting to stretch their dollar even more. Second is Amazon, specifically Amazon same-day delivery. It's very convenient. It's eating, starting to really eat into Target's lunch. And lastly, TikTok Shop. Instead of going to Target to browse, consumers are doing it now through their phone on TikTok Shop specifically for products like candles, mugs, beauty, room decor, dupes of Stanley Cups, not the hockey, but the actual cups, beauty gadgets, budget kitchenware. These are all categories that Target used to dominate in store. TikTok Shop is really starting to eat their lunch. It sounds like there's a money-making opportunity here. So sell Target shares, buy Dollar General, buy Amazon, and if you run a business, find a way to start promoting it on TikTok Shop. Give me a follow or check out my sub stacks linked in my bio if you want more free signals like these.

Creation Date: 2025-05-28 22:04:13+00:00
Video Description: Is XRP like Aluminum? Everyone talks about Bitcoin as digital gold — but what about the rest of the crypto ecosystem? In this video, I break down how each major token maps to the metals and materials that built the modern world: 	•	XRP = Aluminum 	•	Ethereum = Copper 	•	Solana = Titanium 	•	Polygon = Steel 	•	Chainlink = Fiber Optics 	•	Filecoin = Concrete 	•	Arbitrum = Rebar This isn’t financial advice — it’s a new lens to understand what crypto is actually for. If you’re tired of meme coins and price predictions, this one’s for the builders. 🔔 Follow for more crypto breakdowns, analogies, and macro takes. 💬 Drop a comment: Which “metal” are you stacking? #investing #money #crypto #cryptocurrency #xrp #bitcoin #ethereum #financialfreedom #creatorsearchinsights 
Video Duration: 95.0
Number of Likes: 3118.0
Number of Shares: 337.0
View Count: 189700.0
Number of Saves: 798.0
Number of Comments: 277.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.023879810226673695
Tagged Users: 
Hashtags: investing, money, crypto, cryptocurrency, xrp, bitcoin, ethereum, financialfreedom, creatorsearchinsights
Video Transcript: XRP is a new hot crypto token. Why do we need so many tokens? If you're a skeptic, this analogy helps me see things a lot more clearly, including how big of an opportunity crypto really is. First, I want you to picture all the major crypto tokens as if they were metals. Bitcoin is gold, a precious metal. XRP is aluminum, a base metal. It's not flashy. It's not rare. But just like aluminum, it helps move things around very quickly. Money settles within three to five seconds. So it's kind of like Swift, you know, for wiring money. Then you might be asking yourself, as I did, well, like, why would you want to hold wire money through something that moves up and down? XRP settles money so fast that your exposure is very, very, very limited to the actual underlying token. Then think of Ethereum kind of like copper. It powers the whole crypto economy. It is everywhere, just like copper. Copper's in your walls, in your phone, your car. It's the essential infrastructure layer or metal that makes everything run. Then we have Solana, which is kind of like titanium. It's just lighter, more meant for consumer applications. Like when you golf, you don't want a heavy golf club. You want titanium. You don't want a heavy metal. So if you think about the crypto industry more on the use cases, things make a lot more sense instead of all these like coins going up and down. As a former skeptic, I then asked myself, OK, but like, why do you need these coins then like moving up and down? Remember that gold, copper, aluminum, the metals industry all have markets, too, where you can bet on them going up or down in price. There's some other tokens that are worth mentioning that also provide a lot of utility. I just rolled out a bit more of a detailed post on this on my Substack for free, where you can learn a bit more on all of these tokens and what they can do.

Creation Date: 2025-05-27 21:46:08+00:00
Video Description: 05/27: Nvidia chart is flashing signals… but will earnings deliver? MACD + RSI are bullish, but guidance is everything. This is the stock to watch tomorrow.  #stocks #stockmarket #investing #money #financialfreedom #ai #creatorsearchinsights 
Video Duration: 93.0
Number of Likes: 1893.0
Number of Shares: 158.0
View Count: 57100.0
Number of Saves: 188.0
Number of Comments: 118.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.0412784588441331
Tagged Users: 
Hashtags: stocks, stockmarket, investing, money, financialfreedom, ai, creatorsearchinsights
Video Transcript: Tomorrow's a very important day for the markets. NVIDIA is reporting they are 6% of the S&P 500. Options makers have priced in a 7% move in either direction. The cost to run AI models has also come down significantly, 80% over the last two years. What does this all mean? How's the stock gonna do? Here are my two cents. First, let's look at the chart. This is the daily on the stock, and you'll see that it is consolidating. Usually when you see this type of triangle, it means the stock will either jump up or down once it reaches its apex. If you look at momentum indicators, you'll see that both MACD and RSI are showing that the stock is currently overbought. However, for this earnings, I think we actually should be looking at the weekly because it does tell a bit of a more medium to longer term trend, and you'll see it's telling a very different story. Since a lot of the news has come out recently on the company doing a lot of work internationally, you'll see it has reversed from bearish to bullish. A lot of it's gonna come down to guidance, and if you put yourself in the shoes of the CEO Jensen, I think there's incentive to provide very strong guidance, positive guidance, even though there are risks on the geopolitical front, and I would wanna do it to keep the momentum going. You know, a lot of top news and headlines have come out on the company doing a lot of work internationally. Doing so would also scare away competitors, strengthen leverage with suppliers, and it would really give big tech companies really an incentive to keep spending. Here's a free playbook on their earnings I just published on Substack, where I also talk about the impacts of lower compute and inference costs on NVIDIA long.

Creation Date: 2025-05-19 01:06:14+00:00
Video Description: ‼️05/18: How big will markets sell off because of the US debt downgrade? Doing a live tomorrow morning, link in bio 🙏🏻 #stock #stockmarket #money #investing #financialfreedom #creatorsearchinsights 
Video Duration: 59.0
Number of Likes: 1718.0
Number of Shares: 102.0
View Count: 47900.0
Number of Saves: 169.0
Number of Comments: 221.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.04613778705636743
Tagged Users: 
Hashtags: stock, stockmarket, money, investing, financialfreedom, creatorsearchinsights
Video Transcript: Stock market futures are down, and there's a lot of negativity around the Moody's U.S. downgrade. I think people are overreacting a bit here, and here's why. First, Moody's was just very, very slow here, because the S&P downgraded the U.S. already in 2011, and then Fitch downgraded the U.S. in 2023. I posted a video last week saying we could be in a new bull market. You can check out the video in the playlist here below, and my conviction in that continues to build. The biggest drivers are, one, global liquidity. There's over $8 trillion sitting on the sidelines, ready to plow into the stock market. Two, savings. Savings rates in the U.S. continues to go up. I think that'll be a really good buffer in case of any tariff price shocks. Three is strong corporate earnings. Company earnings have been a lot stronger than expected, and this is largely due to companies having spent the last few years preparing for this moment through making it through a global pandemic, high interest rate environment. Lastly, AI. In the first quarter alone, there was a 71% jump in spending on AI infrastructure.

Creation Date: 2025-05-14 15:09:31+00:00
Video Description: 05/14: A new Bull Market is here💰Here’s how to profit. It’s like 1995 all over again 🕺🏻 #stocks #stockmarket #money #investing #financialfreedom #creatorsearchinsights 
Video Duration: 101.0
Number of Likes: 8967.0
Number of Shares: 900.0
View Count: 222600.0
Number of Saves: 1755.0
Number of Comments: 1127.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.05727313566936208
Tagged Users: 
Hashtags: stocks, stockmarket, money, investing, financialfreedom, creatorsearchinsights
Video Transcript: We might be at the start of a new bull market. Here's why and the mindset shift you're going to have to make fast to profit from this. The US-China trade truce was the final data point the market needed to be able to start looking into the future and entering into this new mode. It's the same mode the market was in in 2020. If you guys remember, the stock market tanked and more and more bad news just kept coming out. But the market knew, hey, the Fed's going to backstop, rates are going to be cut to near zero. There's going to be a lot of liquidity in the market, and that's going to drive consumer spending, businesses, and the market. And the market is in a very similar setup right now. So one data point is the yield on the 10-year Treasury is high, but tech stocks, gross stocks are up. Usually when you see this, it's the market saying, look, we know there's probably going to be some risk of inflation, probably less so risk of recession now, but there will be some inflation. But growth is going to offset that. And the market is telling us that it's going to be tech, it's going to be AI. And so what I would be embracing right now, if I was, I mean, I'm doing it, is imagine you're now in the future in 2026. Tariffs are behind us. We have potential tax cuts, deregulation, all really, really good things for the global economy and stocks. And the market is going to start to price that in. The sectors that should outperform in this next wave are here. The big one is AI, because AI is deflationary. It's going to help bring down costs, and a lot of those costs are going to be passed on to the consumer.

Creation Date: 2025-05-13 20:58:10+00:00
Video Description: 💰05/13: The recession is cancelled. Has a new bull market started? #stocks #stockmarket #investing #money #financialfreedom #creatorsearchinsights 
Video Duration: 99.0
Number of Likes: 1908.0
Number of Shares: 145.0
View Count: 56000.0
Number of Saves: 187.0
Number of Comments: 326.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.04582142857142857
Tagged Users: 
Hashtags: stocks, stockmarket, investing, money, financialfreedom, creatorsearchinsights
Video Transcript: Is the recession canceled? That is the question of the day. You'll see that a ton of Wall Street analysts have cut the odds of one happening and even increasing price targets for the stock market. On Calshi, 38% chance of one happening, down from 70. Here's why this is happening so fast, and it has to do with the stock market and consumer confidence. That's where it starts. A higher stock market leads to more consumer confidence and business confidence. On the consumer side, with lower inflation now expected and a higher stock market, consumers will likely spend more and no longer delay the purchase of big ticket items, like potentially a house. And so with more consumer spending and lower inflation, businesses will also benefit. They will expect lower input prices because of lower potential inflation, which will increase their margins in which they can use that money to hire more, which will lead to more profits. And more profits leads to higher earnings per share and stock prices are derived by earnings per share. And with higher stock market, consumers will then also feel better and it is a bit of a vicious cycle. On top of all of this is AI. I feel like AI is going to play a very big role in decreasing inflation because AI will reduce the cost of producing goods and services. AI should make companies, people, governments more productive. And I do think the stock market is starting to price that in. Certain stocks, ETFs and assets will outperform others in a very big way during periods like these. I'll be speaking to that tomorrow morning on a TikTok live. The link is in my bio. But back on the trading floor days, we call this risk on mode and it definitely does feel like that right now.

Creation Date: 2025-05-12 17:17:24+00:00
Video Description: 💰05/12: How to profit from the US / China news - a major rotation is happening fast.  #stocks #stockmarket #investing #money #financialfreedom #creatorsearchinsights #greenscreen 
Video Duration: 89.0
Number of Likes: 1530.0
Number of Shares: 95.0
View Count: 42600.0
Number of Saves: 265.0
Number of Comments: 75.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.04612676056338028
Tagged Users: 
Hashtags: stocks, stockmarket, investing, money, financialfreedom, creatorsearchinsights, greenscreen
Video Transcript: Here's how you can profit from the wild rotation that's happening because of today's U.S. China news. The U.S. dollar has fallen substantially since beginning of the year. You can see here it is now starting to rebound as more confidence and stability builds back on the U.S. economy. A stronger U.S. dollar is not going to help gold. As you can see, the markets are up while gold is down. In the near term, we should also see oil prices rebound a bit here. This news should also help semiconductor and AI stocks. The blue line being SMH, a semiconductor ETF. The red line being the S&P 500. As you can see here, the SMH ETF is now significantly outperforming the S&P. There will be a lot more capital coming back to the U.S. now. You can see here that the red line above is the S&P. The blue line is the European ETF IEV. For some time, the blue has been outperforming the red. But look, as of today, the S&P is now beating the European ETF again. And the same goes for Japan. For quite some time, the red line again being the S&P 500. The blue line being the Japanese ETF EWJ. It is now back to par. In the near term, we're also likely going to see a rotation back into risk-on categories and sectors. As you can see, this is as of today. Consumer staples and utilities are selling off while consumer discretionary, IT, tech, risk-on categories are outperforming. A list of all the stocks I'll be buying and selling can be found on my sub stack. The link is in my bio. Here's a summary of everything I just said and a list of ETFs, sectors, and assets you may want to get in and out of given this news.

Creation Date: 2025-05-11 22:23:20+00:00
Video Description: 05/11: Here’s where the stock market may go next based on history 🧐 Do you agree or disagree? Please leave a comment 🙏🏻 #stocks #stockmarket #investing #money #financialfreedom #creatorsearchinsights 
Video Duration: 110.0
Number of Likes: 1225.0
Number of Shares: 88.0
View Count: 32100.0
Number of Saves: 130.0
Number of Comments: 117.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.048598130841121495
Tagged Users: 
Hashtags: stocks, stockmarket, investing, money, financialfreedom, creatorsearchinsights
Video Transcript: Stock market's behaving a lot like 2008, which means we're gonna have a massive correction soon. Wait, actually, nevermind. It's behaving a lot like 2011 and 2012. Wait, nevermind, it's behaving a lot like the late 90s. There are so many of these comparison charts I'm seeing right now. The fact that there are so many different years that people are comparing the current situation to means people have no idea what's gonna happen next. Here's what we do know, though. We can learn from the past. I think we're closer to a combination between 2018, 2022, and late 90s. So in 2018, market was on a tear until the tariffs actually got implemented and were in market. That is when the stock market actually sold off. In 2022, an inflation scare. The Fed used the same words. Inflation is transitory. Look what happened. Once inflation started to show, it spooked the market and the market sold off. In the late 90s, we had the internet. It was the dot-com era. Right now, we have the AI era. And what I don't think the market is yet seeing is the potential deflationary impacts of AI. If AI, and it's moving super fast right now, if it can truly help companies reduce costs, drive revenue, that's deflationary. If AI can help reduce costs, those costs should be passed off to the consumer. The prices of products and services globally would come down. Workforces become more efficient. Companies become more profitable. So there could be a world whereby the stock market could potentially keep pumping like it did in the late 90s based on AI. And where it's different this time around is there are only 30 million people on the internet back in the late 90s. There are a lot more than 30 million people using AI right now. Love to hear your thoughts. Are these the years we're closest compared to? Are there different years I'm missing? Please leave a comment.

Creation Date: 2025-05-09 17:27:17+00:00
Video Description: Replying to @sbps3000  05/09: The stock market rally may be stalling. It’s time to take some profit 🙏🏻 #stocks #stockmarket #investing #money #financialfreedom #creatorsearchinsights #marketnews #greenscreen 
Video Duration: 69.0
Number of Likes: 2056.0
Number of Shares: 239.0
View Count: 59200.0
Number of Saves: 242.0
Number of Comments: 336.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.0485304054054054
Tagged Users: sbps3000
Hashtags: stocks, stockmarket, investing, money, financialfreedom, creatorsearchinsights, marketnews, greenscreen
Video Transcript: The stock market rally is losing steam fast. I would definitely consider taking some chips off the table today. We're back to pre-liberation day levels, which is wild. The market's about to hit not just one major wall of resistance, but two. How do average people take profits or take chips off the table? You just have to sell some stock. You don't have to sell everything. But as someone who has been in the market now for over 17 years, seen three downturns, I have never regretted taking some chips off the table in a moment like this. Here's another way to think about it. And I'll continue with this poker analogy. So all these Wall Street analysts have decreased their targets for the S&P 500. The S&P 500 right now is at $56.57. The average of all these analysts' price target, the average is $58.08. That's only 2.7% increase from here. So the analogy would be like, it's like going all in when the pot is very small. Is it worth it to make an additional, call it 2.7%, to potentially lose a lot more of that because tariffs have not even started being in market yet?

Creation Date: 2025-05-08 19:41:04+00:00
Video Description: 05/08: Greed is driving this stock market rally and it usually doesn’t end well.  What happens next? Going Live tomorrow morning, link in bio ✌🏻 #stocks #stockmarket #investing #money #financial-freedom #creatorsearchinsights 
Video Duration: 87.0
Number of Likes: 13500.0
Number of Shares: 1317.0
View Count: 408600.0
Number of Saves: 1822.0
Number of Comments: 744.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.04254282917278512
Tagged Users: 
Hashtags: stocks, stockmarket, investing, money, financial, creatorsearchinsights
Video Transcript: Why is the stock market still going up? Didn't the Fed say yesterday that there's increased risk of recession and more inflation coming? Stock market's in full green mode. Here's what's happening and it doesn't usually end well. There's an old saying on Wall Street, buy the rumor, sell the news. And that's exactly what's happening right now. Markets start to price in the anticipation of an event. When the event actually then happens, markets usually would then sell off. So think about, for example, when, I don't know, Tesla was added to the S&P 500. I remember very clearly because I was trading Tesla back then, this was a few years ago. And people would just, and a lot of it is retail traders, buy Tesla, buy Tesla, buy Tesla. If you think about it, nothing fundamentally has changed about the company. But a lot of this buying is greed and it is a lot of retail investors. The difference between this Tesla example and what is happening with tariffs right now is that tariffs actually will have a substantial impact on the economy. So the market had a big sell off a few weeks ago because it was very unexpected what happened. And now as these new trade deals get negotiated, oh, it's better, oh, it's better. And so the market may keep going up, driven by a lot of greed. Once these tariffs get enacted, once real hard data comes out on the impacts, that is when I think the market's going to really sell off.

Creation Date: 2025-05-08 01:38:10+00:00
Video Description: 05/07: Oil prices are tanking..does that mean we’re headed for a recession? It could result in something even worse.  #stocks #stockmarket #investing #money #financialfreedom #economy #crash #creatorsearchinsights 
Video Duration: 173.0
Number of Likes: 13400.0
Number of Shares: 702.0
View Count: 372300.0
Number of Saves: 1197.0
Number of Comments: 847.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.0433682514101531
Tagged Users: 
Hashtags: stocks, stockmarket, investing, money, financialfreedom, economy, crash, creatorsearchinsights
Video Transcript: prices are tanking, isn't that a good thing? No, it's not. It usually signals a recession when that happens, and that's not even the scariest part. So how does this all happen? So firstly, oil prices are tanking because OPEC decided to increase supply. And when that happens, and there's not a lot of demand, prices fall. Prices have to fall because you're not at the equilibrium point. You know, econ 101, supply and demand. And why is there not a lot of demand for oil? Well, there could be many reasons. Slowdown in economic activity, slowdown in shipping, manufacturing, just general slowdown in economic activity. And so during recessions and global slowdowns, oil does fall, and that's why it has spooked the market. And when I said this is not the scariest part, what do I mean? If this continues, this will create another problem for the Fed to fight. We've got inflation, we've got stagflation, and then there's now a potential risk of deflation. Deflation happens when consumers delay, consumers and businesses, delay purchases of goods and services because they expect prices to be lower. And because they expect oil prices to be lower and therefore things get cheaper. That usually is not the case. But as a result, businesses may have to lower their prices and it becomes a bit self-fulfilling. Consumer demand softens, businesses have to lower prices, and therefore corporate profits fall. And another problem for the Fed to fight. And that is why when the Fed spoke today, Jerome Powell has been very careful with his words. Because consumers have already started panic buying ahead of tariffs, pulling up their spending. Spend, spend, spend, spend, spend. If the Fed came out and said the economy is slowing, consumers may delay their spending because they may expect prices to decrease. And if that happens, that'll make things even worse. And that'll drive inflation to come down even faster, making the situation that we're currently in even tougher to get out of. I'm really trying not to be doom and gloom. I'm just trying to relay the bigger picture of what is happening here, why I think it's going to be very important to be patient, why we've only seen the tip of the iceberg, and why this could be a really good time to park money in safe haven assets like Bitcoin, gold, and even international markets. You know, the U.S. is not the only stock market you can invest in. I'll be putting up some more content on other stock ideas from other countries, Europe, China, etc. But please be patient and be careful out there.

Creation Date: 2025-05-04 23:48:14+00:00
Video Description: Replying to @arellehug  ‼️05/04: The stock market may sell off this week based on this real-time consumer spending data.  What should we do? Join me Live at 8:30am EST (link in bio) ✌🏻 Not Financial Advice! #stocks #stockmarket #investing #money #financialfreedom ##creatorsearchinsights#greenscreen 
Video Duration: 98.0
Number of Likes: 8468.0
Number of Shares: 560.0
View Count: 187600.0
Number of Saves: 993.0
Number of Comments: 513.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.05615138592750533
Tagged Users: arellehug
Hashtags: stocks, stockmarket, investing, money, financialfreedom, creatorsearchinsights, greenscreen
Video Transcript: The stock market's about to hit a wall based on real-time consumer spending data. It's the same data that Wall Street's top hedge funds use, the same data we use to predict the first market drop. It's showing that consumer spending fell across all major categories. This will start to get priced in this week. A lot of funds will start selling. Here's what it is showing. Visa is widely viewed on Wall Street as a good barometer of how the economy is doing. You'll see here that since beginning of the year, spending on Visa cards has gone up, but it has hit a bit of a plateau as of recently and looks like it's spiking back down. To verify what I was seeing, I want to look at MasterCard as well to make sure it wasn't just a Visa thing. Same thing. It has plateaued and on its way down. I had my data science team dig in and it is very interesting what we're seeing. So up here is overall spend across debit cards, credit cards. So it's Visa, MasterCard, Amex, and Discover. You'll notice the spike up in spend in March. It is now public knowledge that this is consumers pulling up their spending. So in advance of tariffs, a lot of spending happening. It is very important to note that furniture, fixtures, and appliances has been pulling everything up. So yes, in the month of March, as you can see here, consumer spending across all the major categories saw a pretty big lift. But look at April. It is reversed in a pretty big way. The only category that is seeing a lift is still furniture, fixtures, and appliances. These are expensive, durable goods that consumers are buying because they are concerned that it'll spike up even more in price. It looks like the selling has already started based on stock market features. I'll be doing a TikTok live in the morning Eastern to go through all of this, what it means, what I'm going to be buying, what I'm going to be selling. It's going to be a very big week in the markets. If you've been waiting to buy into this market, this is going to be your chance.

Creation Date: 2025-05-01 23:15:31+00:00
Video Description: 05/01: There’s a massive rotation out of Gold into Bitcoin secretly happening right now. Here’s why‼️ #investing #money #gold #bitcoin #cryptocurrency #crypto #financialfreedom #creatorsearchinsights #greenscreen 
Video Duration: 78.0
Number of Likes: 6410.0
Number of Shares: 940.0
View Count: 167700.0
Number of Saves: 1340.0
Number of Comments: 264.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.05339296362552177
Tagged Users: 
Hashtags: investing, money, gold, bitcoin, cryptocurrency, crypto, financialfreedom, creatorsearchinsights, greenscreen
Video Transcript: There's a massive rotation happening out of gold in a Bitcoin right now. It's causing a lot of confusion in the markets. I just got some Intel and it could be China. Chinese are off on holiday. That explains why gold is down $100. They're also selling and have been selling for some time. I took a look at some charts and it is not just them. There's a reason why everyone is selling gold right now. All right, guys, this is going to blow your mind. So orange is gold. Blue down here is the US dollar index. I drew a line. As you can see, there's a very clear line of support. Every time the US dollar index bounces or hits the line, look what happens. Gold tanks. This happened in 2012, 2014, 2020, and right now, 2025. What happens when the US dollar goes up? Gold comes down because it is now more expensive to buy gold because it is typically denominated in US dollars. You can see here over the last few months, Bitcoin has been coming down. Gold has been going up. But as of today, Bitcoin is now on its way up and gold is coming down. Institutions are back and they are buying Bitcoin like crazy. Since April 22nd, BlackRock's ETF has amassed over $4.5 billion in inflows. Institutional buying has surged. Bitcoin is seen 40% undervalued. And a lot of this is because a lot of governments and central banks are starting to hold Bitcoin. If you know someone who recently bought Bitcoin or should be buying Bitcoin, tag them in the comments.

Creation Date: 2025-04-30 17:37:06+00:00
Video Description: 04/30: This indicator could help prevent a recession and where it closes today will be critical.  #investing #economy #stocks #stockmarket #money #finance #financialfreedom #creatorsearchinsights 
Video Duration: 137.0
Number of Likes: 2537.0
Number of Shares: 201.0
View Count: 64500.0
Number of Saves: 374.0
Number of Comments: 438.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.05503875968992248
Tagged Users: 
Hashtags: investing, economy, stocks, stockmarket, money, finance, financialfreedom, creatorsearchinsights
Video Transcript: Everyone is freaking out on the GDP number that just came out thinking a recession is imminent. There's an indicator that could actually help get the US out of recession and where it closes today is going to be very important. This indicator is a yield on the 10-year treasury. It is at a critical level today. If it breaks it and keeps falling, there could be a world where tariffs get reversed and the Fed next week could even be in a position to start bringing rates down because it is very clear that the economy is slowing. So here's why today is critical. You'll see that since September, the Trump pump brought yields up. That's because people sold treasuries, pumped into the stock market. Since then, it tried to come down. And then on this day was when Trump made the announcement that paused tariffs. That should have brought yields down, but it actually spiked it back up. You'll see though that it has since come back down and we are at this key technical level, this black line. If it breaks it, it is likely to keep falling. And you'll see if you look at MACD on momentum and RSI, both momentum indicators, they are both saying that the yield on a 10-year treasury actually should keep falling. Here's how it would all play out, but it would have to happen very quickly. So if yields continue to fall, there's just less need for tariffs because you're solving for a budget deficit. And you do so by either bringing in more revenue or decreasing your costs. And given the magnitude of the interest payments the U.S. has to make, every 0.5 percent drop in yields is massive. And so if that happens, then there's less need to tariff countries because tariffs are actually terrible, but it feels like it was needed to get revenue very quickly. It is short-term revenue for a country, but medium to long-term, it's terrible because countries won't want to work with you and your overall revenue comes down. Also, what would happen is less inflation because of less tariffs. Jerome Powell next week could be in a position, again, if this all happens, to potentially consider decreasing rates, cutting rates, because the only reason why he's not doing that right now is because of the inflationary impact of tariffs. And if he were to do that and signal more rate cuts ahead, the yield would fall even more so, making the interest payments even less. I've just been sitting here thinking about this, I'm like, am I crazy? Would love to get your comments and thoughts on this. And am I missing something?

Creation Date: 2025-04-29 21:04:21+00:00
Video Description: 04/29: Impacts from tariff talks are getting to concerning levels. Here’s how to protect your portfolio.  #investing #money #stocks #stockmarket #financialfreedom #creatorsearchinsights #greenscreen 
Video Duration: 97.0
Number of Likes: 566.0
Number of Shares: 96.0
View Count: 16000.0
Number of Saves: 52.0
Number of Comments: 106.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.05125
Tagged Users: 
Hashtags: investing, money, stocks, stockmarket, financialfreedom, creatorsearchinsights, greenscreen
Video Transcript: US and China need to figure things out soon. Some are saying that it's gonna be very hard to reverse. We are starting to see the damage. Port volumes in LA are starting to plummet. Here's what I think could happen next. The most immediate and direct impact is gonna be chaos in the world of retail. A lot of companies loaded up on products and inventory over the last few weeks in anticipation of these tariffs. We haven't seen much impact on prices yet. And the key is yet. Because many of these companies, it was business as usual. Analysts are saying that in the next five to seven weeks, a lot of the inventories that companies have bought will be depleted. What will that mean? It'll mean that we'll likely see more of this, emptier shelves. A secondary impacts that will be even more harmful, which is increased prices. It's also gonna be very hard for retailers to forecast, which may eat into corporate profits, especially for big retailers like Walmart, Target, and even Amazon. We're also likely gonna see economic slowdown, especially at key ports. Key ports being Los Angeles, Long Beach, Tacoma, Oakland, and Seattle. 60%, 6-0% of their imports are tied to China. As imports collapse, this will have trickle-down effects. Not only are the dock workers, warehouse workers, truck drivers impacted, it will flow into the local economy. First, this may seem like a small localized thing, but when you add it all up, that is where this could have pretty major impacts on the full economy. First two impacts are more recessionary. Supply chain disruptions is gonna be more inflationary. It's just gonna be more expensive for companies to source goods elsewhere. And those costs are likely gonna be passed onto the consumer. Hopefully this drives some urgency for both sides to figure things out. Until then, I continue to encourage people to hold assets like gold and Bitcoin that usually do well in times of uncertainty like.

Creation Date: 2025-04-29 15:09:39+00:00
Video Description: 📈AI Stock Ideas 2025: Thumbs Up Media (TZUP) Disrupting the $148b Advertising industry with AI agents to promote brands Not Financial Advice. Please do your own research 👊🏻 #investing #stocks #stockmarket #money #financialfreedom #creatorsearchinsights 
Video Duration: 85.0
Number of Likes: 945.0
Number of Shares: 83.0
View Count: 23100.0
Number of Saves: 181.0
Number of Comments: 57.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.05480519480519481
Tagged Users: 
Hashtags: investing, stocks, stockmarket, money, financialfreedom, creatorsearchinsights
Video Transcript: Here's a trending AI stock I want to tell you guys about. The company is called Thumbs Up Media, ticker TZUP on the NASDAQ. The company IPO'd only a few months ago. The current market cap is under 50 million, making it a small cap play with a ton of potential upside. $148 billion. This is the total amount of money spent on display ads in the US. Also the market that Thumbs Up Media is looking to disrupt. The advertising industry is still pretty archaic. Yes, it has gone through a ton of transformation, but the way that businesses reach consumers today is pretty outdated. And it's done through ads. It's done through clicks and eyeballs and in a very static way. Thumbs Up Media is looking to change all of that by deploying AI agents to promote products, brands, and services in a lot more of a personalized one-to-one way. The company is growing a ton and building a lot of very interesting technologies. As you guys know, I only talk about and promote companies I believe in, and this is one of them. The big why now, right now, for a company like Thumbs Up Media is that marketing has become very expensive. It is very costly for a business to reach you and I. And I do believe that this space is going through a lot of transformation right now. The future of the advertising industry is gonna be a lot more personalized and one-to-one because at the end of the day, who likes ads? So using AI to make marketing a lot more cost-effective is gonna be very top of mind for marketers, especially during an economic climate like today. This video is in partnership with Digitallandia on behalf of Thumbs Up. This reminds me a lot like the early days of social media. So there are a lot of opportunities ahead and Thumbs Up is right in the middle of it.

Creation Date: 2025-03-07 21:40:47+00:00
Video Description: Replying to @mollychen181  8 stocks to buy to protect AND grow your portfolio in a downturn: Costco Walmart Marathon Peteoleum Coca Cola Nvidia Marvell Technologies Broadcom Tesla #stocks #money #investing #financialfreedom #market #marketnews
Video Duration: 87.0
Number of Likes: 114700.0
Number of Shares: 26700.0
View Count: 2200000.0
Number of Saves: 73634.0
Number of Comments: 2101.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.09869772727272727
Tagged Users: mollychen181
Hashtags: stocks, money, investing, financialfreedom, market, marketnews
Video Transcript: Which stocks should we buy in a recession? Here are eight stocks I'm personally rotating into in my million dollar portfolio in case the economy slows. As I mentioned in the prior video in this playlist, half my portfolio are going to be low risk stocks focused on protecting capital. And the latter half of the portfolio are going to be more growth oriented stocks that should still do well in a slowdown type of environment. The low risk side of the barbell portfolio, the first name is Costco. Consumers will be buying in more bulk as the economy potentially slows. Second company is Marathon Petroleum. If you have a car, you're going to need gas. Spending data we have access to is showing pretty good growth on the name as well. Maybe more consumers are doing staycations right now and driving instead of flying. Number three is Walmart. It's a name that typically does well during slowdowns and a name that a lot of folks, including large institutions, rotate into. Number four is Coca-Cola. You know, it's a name everyone knows, strong balance sheet, and it's a consumer staple. On the high risk side, these are all growth stocks with AI exposure. I think corporations are going to continue to invest in AI heavily, even in a slowdown type of environment. Four stocks here are Nvidia, Broadcom, who had very strong earnings today. Marvel Technologies, a name that's recently been beaten up, so it could be a good entry point right now. The last one is Tesla. I know it's a bit controversial, but in a slowdown type of environment, consumers may trade down to EV because it is cheaper. I'm starting to build up my positions in this portfolio, so follow along if you want tips and tricks on how to properly enter and exit into stocks.


Example Finfluencer Profile 2:
- Profile Image: https://p16-sign-va.tiktokcdn.com/tos-maliva-avt-0068/7312923100926967854~tplv-tiktokx-cropcenter:720:720.jpeg?dr=14579&refresh_token=7183c5a6&x-expires=1748991600&x-signature=hsmS0YMPdA4TGnFAt%2BogtcALHus%3D&t=4d5b0474&ps=13740610&shp=a5d48078&shcp=81f88b70&idc=maliva
- Profile Name: humphreytalks
- Profile Nickname: Humphrey Yang
- Profile Biography: humphreytalks@gmail.com
Personal Finance and Investing Videos
Templates and YT↙️
- Profile Signature: humphreytalks@gmail.com
Personal Finance and Investing Videos
Templates and YT↙️
- Profile Biography Link: https://beacons.ai/humphreytalks
- Profile URL: https://www.tiktok.com/@humphreytalks
- Profile Language: en
- Profile Creation Date: 2019-11-10T17:25:41.000Z
- Verified Status: True
- Number of Followers: 3400000.0 Followers
- Following: 499.0 Users
- Total Number of Likes: 56100000.0
- Total Number of Videos: 1122.0
- Total Number of Digg: 0.0
- Private Account: False
- Region: US
- TikTok Seller: False
- Average Engagement Rate: 0.0100165818759936
- Comment Engagement Rate: 0.0001317567567567
- Like Engagement Rate: 0.0098848251192368
- Video Transcripts (Sorted from Newest to Oldest):
Creation Date: 2025-05-12 18:02:53+00:00
Video Description: The US and China have agreed to a trade deal when it comes to tariffs. Here's what it means for you.
Video Duration: 66.0
Number of Likes: 862.0
Number of Shares: 53.0
View Count: 25300.0
Number of Saves: 44.0
Number of Comments: 94.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.041620553359683794
Tagged Users: 
Hashtags: 
Video Transcript: The US and China have agreed to a trade deal when it comes to tariffs, and here's what it means for you. So first, the deal means that reciprocal tariffs between both the countries go from 125% to 10%, that's really good news, but the US will still keep a 20% extra duty on fentanyl in place, meaning that the total tariffs on China stand at around 30%. The deal will be for 90 days as they continue to negotiate for a longer term plan, but it still should mean a few things. First is that goods imported from China to the US won't incur these crazy tariff rates, which hopefully means that these costs aren't going to be passed down to you. Second is that the risk of recession in 2025 has decreased to 39% according to Polymarket, but the long-term future is still uncertain. Number three is that quarter one GDP came in negative, which is usually a sign of a contracting economy, but that was largely due to the fact that there were so many imports coming in. Many companies were importing goods like crazy to front-run tariffs, and imports actually subtract from GDP, so we'll actually see if that changes for Q2. And number four, stocks should recover barring any catastrophic events and should continue to do fine for at least the next 90 days. Let me know if you have any questions in the comments.

Creation Date: 2025-05-06 18:15:58+00:00
Video Description: My Coca Cola Dividend! Do you guys think it’s worth it? 
Video Duration: 65.0
Number of Likes: 6456.0
Number of Shares: 93.0
View Count: 118700.0
Number of Saves: 351.0
Number of Comments: 144.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.05934288121314238
Tagged Users: 
Hashtags: 
Video Transcript: So the other day I was checking my investment account and I received $12.60 from Coca-Cola as a dividend. That's a percentage of profits that Coke passes back to me every quarter just for owning their shares. Now, I realized that there are a few reasons Coke does this. They're a dividend king, which means that they've been paying dividends for at least 50 years. And in Coke's case, they're going on 62. Paying a dividend is also a conscious decision that shows to the market that they are a stable company because they have enough profits to pass back to their holders. And it is said that Coca-Cola is one of the best defensive companies to own during recessions and bear markets because people are going to drink Coca-Cola no matter the economic environment. But the story doesn't end there because Coca-Cola is owned by a lot of big investors for the passive income in dividends it provides. Warren Buffett's Berkshire Hathaway, for example, now owns roughly 9% of Coca-Cola, which pays them approximately $815 million every single year in dividends. But if you or I wanted to make just $100 a month from Coke in dividends, well, you would need about 588 shares or about $42,402 worth of Coca-Cola stock. So those dividends do come at a hefty price.

Creation Date: 2025-04-07 15:43:18+00:00
Video Description: The S&P is officially in correction territory and flirting with bear market territory after the Liberation Day Reciprocal Tariffs were announced last Wednesday. Here’s what I would be doing. 
Video Duration: 65.0
Number of Likes: 3709.0
Number of Shares: 164.0
View Count: 84100.0
Number of Saves: 335.0
Number of Comments: 179.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.0521640903686088
Tagged Users: 
Hashtags: 
Video Transcript: The stock market is now officially down over 15% in the past three trading days due to the reciprocal tariffs. There's a lot of uncertainty and volatility right now, so what should you be doing in times like these? I have three action items. Number one, don't get emotional. If you're a passive investor with a 401k, IRA, index funds, and you're investing for the longterm, this is not the time to be panic selling. I personally know a lot of people that sold during the COVID crash, and they basically regretted it because the market recovered within two or three months, and they missed out on a ton of gains. If you're investing for the longterm, there's a very probable chance that the market is going to be way higher later on, so you just wanna stay the course. Number two, if you do wanna buy into the market, make sure to average into it. So perhaps you choose certain levels, like if the S&P 500 hits 5,000, 4,800, 4,600, or 4,400, you can average in that way. Or the simplest way to average in and without the regret of buying too high or too low is just to average in every certain number of days or weeks. And number three, it can be really tough looking at your portfolio every day, so you just really wanna focus on your main source of income and don't lose sight of that.


Example Finfluencer Profile 3:
- Profile Image: https://p16-sign-useast2a.tiktokcdn.com/tos-useast2a-avt-0068-euttp/b549b3b86b42a254db09fefbd0a5c34a~tplv-tiktokx-cropcenter:720:720.jpeg?dr=14579&refresh_token=3d8d310a&x-expires=1748991600&x-signature=w3W%2B055mL4a2FbIcLSkoZQuhyxw%3D&t=4d5b0474&ps=13740610&shp=a5d48078&shcp=81f88b70&idc=my
- Profile Name: financialtimes
- Profile Nickname: FinancialTimes
- Profile Biography: Stay ahead, follow the FT
- Profile Signature: Stay ahead, follow the FT
- Profile Biography Link: https://www.ft.com/register?segmentid=486333ad-cc96-e0dc-de4b-b14e86eb6d59
- Profile URL: https://www.tiktok.com/@financialtimes
- Profile Language: en
- Profile Creation Date: 2024-02-23T15:46:14.000Z
- Verified Status: True
- Number of Followers: 163400.0 Followers
- Following: 6.0 Users
- Total Number of Likes: 1300000.0
- Total Number of Videos: 573.0
- Total Number of Digg: 0.0
- Private Account: False
- Region: GB
- TikTok Seller: False
- Average Engagement Rate: 0.0119617065920615
- Comment Engagement Rate: 0.0020500087427872
- Like Engagement Rate: 0.0099116978492743
- Video Transcripts (Sorted from Newest to Oldest):
Creation Date: 2025-05-30 07:42:20+00:00
Video Description: Margo has amassed a very healthy retirement fund, in part by buying shares in Berkshire Hathaway. But as the investment company reduces its exposure to some US stocks, Margo wonders if it’s time for her to follow his lead and look to invest in other parts of the world. FT’s consumer editor Claer Barrett examines if Margo’s seven-figure portfolio could be particularly vulnerable to a market correction. Tap the link to read more. #BerkshireHathaway #Buffett #WarrenBuffett #Investment #stocks 
Video Duration: 105.0
Number of Likes: 128.0
Number of Shares: 2.0
View Count: 3548.0
Number of Saves: 11.0
Number of Comments: 22.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.04594137542277339
Tagged Users: 
Hashtags: berkshirehathaway, buffett, warrenbuffett, investment, stocks
Video Transcript: Margot, you're fantastically set up for retirement. You've built up a seven-figure portfolio, and one of your biggest holdings, around a quarter of that, is Berkshire Hathaway. Now, tell us why you picked that and a bit about what this stock is. So, it's a firm that's run by an investor called Warren Buffett. Yep, I think we've heard of him. What I like about the investment philosophy is just the idea of looking for businesses that have inherent value. I think that Warren Buffett and his investment partners have done very well in finding those businesses. But I sense a but is coming. I had been concerned for some time about having such a big exposure to the Berkshire Hathaway position. That investment model seems to have found its ceiling. Now, Buffett fans will know that a record level of Berkshire's portfolio is held in cash or bonds, nearly a quarter of its overall value at the last count, as Warren Buffett trims his equity holdings. Plus, he hasn't made any big acquisitions recently, and he's 94 years old. I mean, how does that make you feel as an investor who's held Berkshire Hathaway for many, many years? Yeah, I mean, breaking those two questions down separately, I think maybe the first one's possibly the more important one, i.e. the fact that there's a huge cash pile. That's because he's, my impression is, he and the people he's trained around him are very disciplined in their investment decisions. And they're not going to invest just because they've got a huge cash pile. They're going to invest if they find the right opportunities. And so maybe this tells us it's another information point that says... That US stocks are overvalued? For example, and the fact that he can't find solid domestic businesses to invest in and so on.

Creation Date: 2025-05-29 13:18:48+00:00
Video Description: From food delivery to fast fashion, the option to buy now, pay later is everywhere. But did you know this type of lending is unregulated in the UK? FT's retail banking and fintech reporter Akila Quinio discusses how lenders such as Klarna and Clearpay will soon have to abide by similar rules as a mainstream bank. Tap the link to read more. #Klarna #Clearpay #Buynowpaylater #Bank #Retail #Spending 
Video Duration: 104.0
Number of Likes: 135.0
Number of Shares: 1.0
View Count: 3012.0
Number of Saves: 10.0
Number of Comments: 27.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.05743691899070385
Tagged Users: 
Hashtags: klarna, clearpay, buynowpaylater, bank, retail, spending
Video Transcript: From food delivery to fast fashion, the option to buy now, pay later, is everywhere. Companies like Klarna and Clearpay are letting more and more customers split their purchases into installments without charging interest. But did you know that this type of lending is currently unregulated in the UK? This means that the companies offering these loans do not have to check if consumers can afford them. It also means that borrowers are unable to make complaints to the ombudsman, like they can do with their banks. That's why the UK Labour government is changing this with new legislation which they are bringing forward to regulate the sector. It has been nearly four years since the previous Conservative government first announced plans to do it, so these rules have been long awaited. The government described it as putting an end to the wild west of buy now, pay later, or BNPL. It's good news for consumer groups who have for years criticised these companies, saying they put vulnerable people at risk of accumulating debt from different providers and being hit with late repayment fees. Companies like Klarna have publicly welcomed the rules, saying they will bring clarity and trust that will be good for business. After intense lobbying, they got a big win from the government, which is the fact that they will get a bespoke regime around the disclosures they have to make to customers before they take out a loan. This means they will get their own user-friendly disclosure rules instead of the clunkier ones that were designed 50 years ago under the Consumer Credit Act. Even though consumer groups say affordability checks will help protect consumers, there are still questions around whether people should be using these loans for small everyday purchases if they can't afford them. Now that the government has brought this to Parliament, financial regulators have a year to come up with the detail.

Creation Date: 2025-05-28 15:49:45+00:00
Video Description: Panda bond issuance — renminbi borrowing by overseas companies in mainland Chinese markets — hit Rmb194.8bn ($26.5bn) in 2024, the highest on record for a full year. Analysts say this 'in China, for China' strategy could help foreign groups reduce transaction costs — or even hedge against potential financial restrictions on their local units if the US-China trade war escalates. Tap the link to read more #China #Markets #Economy #Finance #Pandabond 
Video Duration: 58.0
Number of Likes: 277.0
Number of Shares: 17.0
View Count: 8079.0
Number of Saves: 50.0
Number of Comments: 16.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.04455997029335314
Tagged Users: 
Hashtags: china, markets, economy, finance, pandabond
Video Transcript: Multinational companies are borrowing in renminbi at a record rate, as they search for cheaper funding and a way to hedge against US-China relations. Panda bond issuance is renminbi borrowing by overseas companies in the mainland. In 2024, total borrowing was almost 200 billion renminbi, which is the highest on record. In the first quarter of this year, panda bond issuance was more than 40 billion renminbi, the second highest quarter on record. This marks a shift in strategy by global companies to raise debt for their Chinese subsidiaries locally. It was also boosted by Chinese authorities allowing companies to transfer funds raised from panda bonds outside of mainland China. Analysts say this so-called in-China, for-China strategy can help companies reduce transaction costs and hedge against potential financial restrictions if US-China tensions escalate further. Some international organizations and foreign governments like Hungary have also issued panda bonds.

Creation Date: 2025-05-28 10:36:04+00:00
Video Description: US President Donald Trump’s ‘big, beautiful’ bill was passed by the House of Representatives last week. It adds yet more deficit spending to America's budget. But can it survive the bond market? On the latest episode of Unhedged, Rob Armstrong and Katie Martin discuss how deficit spending affects the bond market and what can happen when yields rise dramatically. Tap the link above to listen to the full episode #financialtimes #markets #bonds 
Video Duration: 92.0
Number of Likes: 198.0
Number of Shares: 5.0
View Count: 7335.0
Number of Saves: 19.0
Number of Comments: 17.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.032583503749147924
Tagged Users: 
Hashtags: financialtimes, markets, bonds
Video Transcript: So the bonds have weakened, and as you say, four and a half or so on the 10-year, I'm sure we can live with that, but there's something about... We can, and it should be said that's within the kind of long-term trading range that goes back to 22. But as with so many financial variables, what matters is less the level than the direction of change. Yes. And the direction of change is up for yields, down for bond prices. Also, one level that does sort of matter to people, there is something about the number five that really sticks in bond markets' heads. And the 30-year US Treasury bond is trading with a yield of about 5%. And that tends to be getting close to what people sometimes call the danger zone. And it's not difficult to imagine that at a certain point pretty soon, lots of investors will start saying, hang on, why do I want to be buying stocks when I don't know what's going to happen next, when I could just buy this Treasury bond and Uncle Sam is going to pay me back one day with a yield of 5%? And it's a yield of 5%. And let's say we expect inflation to be about 3% for the next 30 years. That means you're just collecting 2% a year in real terms from the US government. Yeah, just by buying the bond. You don't have to, just by buying the bond. Yeah, at a certain point, the bonds just get too, just too good to turn down.

Creation Date: 2025-05-27 16:28:58+00:00
Video Description: The US dollar has been the world’s leading currency for a century, but will its dominance begin to fade away? Chief economics commentator Martin Wolf discusses if the dollar is being mismanaged under US President Donald Trump. Tap the link to read more. #Dollar #USDollar #currency #america 
Video Duration: 110.0
Number of Likes: 1847.0
Number of Shares: 106.0
View Count: 46000.0
Number of Saves: 276.0
Number of Comments: 127.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.051217391304347826
Tagged Users: 
Hashtags: dollar, usdollar, currency, america
Video Transcript: Could the dollar cease to be the world's dominant currency? The dollar has in fact been in this position for at least 100 years since it replaced the pound sterling. But today people are beginning to ask quite serious questions whether this will continue. The crucial thing to remember is that the dollar's strength is based on very profound advantages. The United States is still the biggest economy in the world. It has open and liquid capital markets. The currency is as a result widely used in trade and finance across the world. People want to use it because everybody else uses it. So how could that change? Well one reason would be that people start thinking the dollar is really being seriously mismanaged. The trade policy of the United States looks pretty frightening right now. Lots of people are worried about that. The fiscal policy is also worrying. The debt ratio, the ratio of public debt to GDP is exploding. All these are reasons why people worry about the dollar. But at the same time, what's the alternative? Well the most obvious one is the euro. But the euro is at the moment much less significant. It doesn't have a liquid deep capital market. There are no eurozone bonds. They're national bonds. Each of those markets are much smaller than America's. It isn't a political union. The alternative is China, which definitely is a powerful economy. But the problem with China is that it has capital controls. The markets are not deep and liquid. They're not accessible for most people in the world. So in the end, the conclusion is simple. The dollar is not as strong as it used to be, but it's better than all the others.

Creation Date: 2025-05-27 10:15:02+00:00
Video Description: The tit-for-tat tariff escalations between the US and China are on pause, at least temporarily. But if the world’s two biggest economies don’t make progress by July, tariffs could return with a vengeance. How can the two parties make progress? And what does China actually want from the US? The FT's Soumaya Keynes speaks to Jay Shambaugh, former under-secretary of the US Treasury for International Affairs, to find out. Tap the link above to listen to the full podcast.
Video Duration: 94.0
Number of Likes: 185.0
Number of Shares: 1.0
View Count: 9325.0
Number of Saves: 14.0
Number of Comments: 25.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.024128686327077747
Tagged Users: 
Hashtags: 
Video Transcript: So it is possible that there will be a mega deal that may even address some of the structural issues that were problems under your time. Do you think that's the most likely scenario? What do you think is going to happen? I think the Chinese have shown, even relative to other countries, that they very much do not want to be seen as caving in to President Trump. I think they were one of the only countries that did any retaliation to the tariffs that were announced in April. They, as I said before, have been very focused on making sure they are ready for a fight like this, if need be. President Xi is very adamant that China is itself a great power and they are not going to be bullied. So I think that may make it hard to reach something that looks like a grand bargain, if just because it's not clear China wants to be seen as crafting a great bargain. I do think a baseline tariff is going to always be there. I think the Trump administration, unlike the first Trump administration, is very focused on the revenue they get from tariffs. And that's one reason you're seeing that baseline of 10 percent against everyone. I think there will probably be some attempt to climb down and there will be basically some window dressing. At the same time, over time, there might be progress on the Chinese side, just because, as I said before, there are things that really are in China's interest. And so I think China will take steps in those directions. I do think some sort of purchase agreement where the president can argue he has cut a deal, I think, is quite likely.

Creation Date: 2025-05-22 09:08:40+00:00
Video Description: The EU is in Donald Trump’s crosshairs. So how should it respond? Paschal Donohoe, Ireland’s finance minister and president of the Eurogroup of finance ministers, joins FT chief foreign affairs commentator Gideon Rachman to discuss how well placed Europe is to respond to the tariff war and whether the euro can challenge the dollar. Tap the link above to listen to the podcast. #FT #FinancialTimes
Video Duration: 87.0
Number of Likes: 82.0
Number of Shares: 1.0
View Count: 2973.0
Number of Saves: 5.0
Number of Comments: 16.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.03498150016818029
Tagged Users: 
Hashtags: ft, financialtimes
Video Transcript: the EU still has to somehow figure out how it's going to respond to Trump, how do you think they should? I'd make the point that for those collection of democracies, to act collectively is the best way of dealing with this challenge and to allocate competence of that to one institution is the most effective way of doing that. And yes, it can therefore mean that we need to spend a longer period of time building up consensus then regarding how we act. But we do, and we are, want to engage with the US on that and see can we create that deal. But the way that Trump seems to like to act is very kind of impulsive, one-on-one, and the EU just can't respond in that way because you've got to go back to Brussels and get consensus of 27 people and so on. But agility isn't always the same thing as speed. Some of the time it is, some of the time it's not. And from the engagement that I'm having with the Commission at the moment and the work that I know they're doing with other Member States, if you do want an agreement, which I hope and believe the US wants, I think it's in our interest to have an agreement that can be durable and that can stand the test of time. And the EU does have the ability to do that. And you are right, it could take a little bit more time because we are a collection of countries as opposed to a single country. Then the economic value of that agreement is far bigger.

Creation Date: 2025-05-21 17:33:36+00:00
Video Description: Tensions in the Arctic are rising, as melting sea ice opens a new maritime passage across the North Pole, triggering a race to access mineral belts exposed by warming oceans. Tap the link to read more. #Arctic #northpole #minerals #army
Video Duration: 95.0
Number of Likes: 129.0
Number of Shares: 5.0
View Count: 3055.0
Number of Saves: 10.0
Number of Comments: 31.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.057283142389525366
Tagged Users: 
Hashtags: arctic, northpole, minerals, army
Video Transcript: Why are people talking about security threats in the Arctic? Even before his inauguration, Donald Trump put the Arctic on the global agenda when he restated his desire to buy Greenland, a territory which belongs to the US's NATO ally, Denmark. The US space base on Greenland is a key part of its early warning system to detect Russian long-range missiles, so the president wants to boost defences there. The reason for increased concern is that Moscow has reopened a string of Cold War military bases along its northern coast, where its most sensitive nuclear assets are stored. Its submarines are venturing down to the Atlantic within range of European capitals in greater numbers, and tensions with NATO countries are rising as a result of Russia's invasion of Ukraine three years ago. Moscow and Beijing are also investing in a new maritime passage, the Northern Sea Route, which will cut the journey time between Asia and Europe. This critical route is becoming more accessible as Arctic ice melts. So does this mean there will be war in the Arctic? No, not necessarily. But the race between superpowers to access minerals buried under the ice and to navigate the Northern Sea Route has sparked a competition for control. Conflict might not start in the Arctic, but it could easily spread to this region from elsewhere in Europe. What is clear is that Russia tests many of its most advanced weapons in Arctic waters and is also trying out its so-called hybrid warfare capabilities in northern Norway, high in the Arctic Circle. These actions, such as cutting cables and jamming aircraft GPS, fall beneath the threshold of formal armed conflict. But they signal rising hostility between Russia and the West.

Creation Date: 2025-05-21 14:41:30+00:00
Video Description: Recent announcements about trade deals with the UK and China have cheered markets. But what exactly did the UK agree, and why? The FT’s senior trade writer Alan Beattie joins Katie Martin to unpack the negotiations. Tap the link above to listen to the podcast. #FT #FinancialTimes
Video Duration: 90.0
Number of Likes: 173.0
Number of Shares: 3.0
View Count: 6983.0
Number of Saves: 11.0
Number of Comments: 25.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.030359444364886152
Tagged Users: 
Hashtags: ft, financialtimes
Video Transcript: I just think it's going to be interesting to see the incredibly methodical, serious trade talk pro-Europeans in a room with whoever it is that the US puts forward. Maybe it's Peter Nabarro, maybe it's Scott Besson, the Treasury Secretary, who knows? Indeed, and if there's any weakness that the EU has, I think it's that they're assuming they're dealing with a more rational interlocutor. One thing I know that EU people have said is they've assembled this list of retaliation and it's done the traditional way people do retaliation, which is you choose products which are going to hurt politically. In the US, that means you go after particular products from particular states. They always used to go after bourbon because that was Kentucky, because that was Mitch McConnell who was Senate leader for a long time. I think this time they've said they're going to go after soybeans because that's Mike Johnson, who's from Louisiana, and soybeans from Louisiana. The problem with this, I think, is I'm not sure Trump really cares what anyone in Congress thinks. He's making trade policy from the hip using these emergency powers and completely ignoring anything that anyone in Congress says. On his nice new plane from Qatar. On his nice new plane. So is he really going to sit there going, oh no, the soybean farmers of Louisiana are slightly angry. I had better stop my trade war. That seems to me slightly improbable. So I'm not convinced the EU is on the right track if it's trying to use the old playbook on that. I'm not convinced the EU is on the right track if it's trying to use the old playbook 

Creation Date: 2025-05-20 17:52:25+00:00
Video Description: US tariffs have sent financial markets into a frenzy in recent weeks, but how much should central bankers be taking trade into account when setting monetary policy? Tap the link to hear more from Bank of England Monetary Policy Committee member Swati Dhingra and her conversation with Soumaya Keynes. #Banks #London #BankofLondon #America #Finance 
Video Duration: 99.0
Number of Likes: 137.0
Number of Shares: 6.0
View Count: 4658.0
Number of Saves: 11.0
Number of Comments: 32.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.0399313009875483
Tagged Users: 
Hashtags: banks, london, bankoflondon, america, finance
Video Transcript: I see why the Bank of England has a trade economist on its Monetary Policy Committee. Could the Trump administration learn anything from a trade economist based in the UK, perhaps based on some recent experiences over the last decade or so? I think we should always learn from big episodes for the reason that they don't happen very frequently and I think the UK's experience with Brexit and how that panned out over the years is really instructive in that regard because we don't have many other examples of when developed countries put on trade barriers with their key trading partners. And I think what we learned from that episode was, you know, one, the effects can be felt fairly quickly. So the sterling depreciation that happened, it happened within a matter of minutes of when the Sunderland vote came out and sterling lost its value by 10% from subsequent years. That, of course, translated into higher import prices for the UK, both for businesses as well as for consumers. As a result of that, we had a three-year period in the UK from about 2017 to 2019 when real wages almost did not grow at all. So I think the key lesson here is that, you know, it's very attractive before it's happened to think that you would be able to reorient, remake yourself, change your position in the world economy. When the time comes, the performance need not be quite as sanguine. Well, I am sure President Donald Trump is taking detailed notes.

Creation Date: 2025-05-16 08:14:05+00:00
Video Description: Donald Trump’s trade policies have put global markets through the mill in recent weeks. But his policies didn’t come from nowhere. Aspects of US protectionism preceded Trump’s second term – and countries across the world have been pushing for greater self-sufficiency for some time. Is this drive for greater self-sufficiency misguided? Is true self-sufficiency even possible? The FT’s senior business writer Andrew Hill sits down with Ben Chu to discuss. Tap the link above to listen to the full podcast. #FT #financialtimes #trump
Video Duration: 89.0
Number of Likes: 88.0
Number of Shares: 4.0
View Count: 2350.0
Number of Saves: 7.0
Number of Comments: 19.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.05021276595744681
Tagged Users: 
Hashtags: ft, financialtimes, trump
Video Transcript: So the US and the UK have just struck a deal to get past the initial tariffs that were imposed or threatened by Donald Trump. What are the risks of the approach that both the US and the UK are taking with this deal? Well, if you think about it on a global scale, we have a system of global rules embodied in the World Trade Organisation going back decades that countries will not discriminate against other countries in terms of the tariffs they impose. So they can have a high tariff or a low tariff, but that has to be the same for everyone who is importing into them or to whom they're exporting, unless there is a comprehensive free trade deal between two countries, in which case they can have lower tariffs. The issue with the UK-US trade deal, not going into the specifics of individual sectors, is that this is not a classic free agreement where they've done a comprehensive deal. But as part of it, the UK has agreed to lower some of its tariffs on imports from America unilaterally. It's not doing the same globally, so other countries will not have those privileges in terms of exporting those products into the UK that the US has. And the danger is, as many analysts have identified, that this is capitulating or conceding the global system in a way which is quite potentially damaging.

Creation Date: 2025-05-15 09:35:07+00:00
Video Description: Lower-than-expected inflation and the start of negotiations with China seemed to help stocks on Monday. But the dollar remained uncharacteristically weak. Rob Armstrong, the FT's US financial commentator and FT financial reporter Aiden Reiter, ask if America is feeling good, or just relieved to be alive. Tap the link above to listen to the podcast. #FT #FinancialTimes
Video Duration: 97.0
Number of Likes: 138.0
Number of Shares: 9.0
View Count: 6283.0
Number of Saves: 12.0
Number of Comments: 18.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.028171255769536847
Tagged Users: 
Hashtags: ft, financialtimes
Video Transcript: It's possible that markets have really seen how terrible it could be with the reciprocal tariffs. And just any walk back from that horrible extreme is welcome news. Also, I mean, we can even think back to his campaign promises. Originally, he said 10 percent around the world, 60 percent on China. Yes. We're below that. Or we're at or below that. Yes. With the potential to go lower. Yes. So if anything, this is a big departure or a slight improvement from what he said earliest phase. 20, 20 percent of the 30 percent that's still on China is the fentanyl. Tax. That looks to me to be ripe to remove. Right. That's something that just screams out. Negotiate me away. But now where are we going to wind up with lower tariffs on China than we had going into this administration? Because, you know, remember, we had plenty of tariffs that Trump had put on and then Biden even added to. If China is clever, they will give Trump enough other concessions of various flavors that he can hold up the trophy and declare victory while bringing tariffs down further. Yeah. I mean, we can give China some praise here, too. When Trump first hit them with tariffs in, what, February and January, they chose very targeted, retaliatory tariffs. Right. They were politically meaningful and less economically meaningful. Right. Tariffing coal, which is mostly produced in states that support Trump. Tariffing hyper specific things that theoretically would bring Trump to the table. I don't know if that's actually what brought Trump to the table, but in retrospect, it seems like a good gambit by them.

Creation Date: 2025-05-14 15:53:29+00:00
Video Description: China has placed export restrictions on rare earth elements and the magnets they go into. China already dominates global reserves and has a little under half of the world’s rare earths reserves, according to the US Geological Survey. Camilla Hodgson discusses the details of the ban and which countries other than the US are being affected. #rareearth #metal #metals #china #export #trade 
Video Duration: 91.0
Number of Likes: 164.0
Number of Shares: 19.0
View Count: 4812.0
Number of Saves: 14.0
Number of Comments: 18.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.04467996674979219
Tagged Users: 
Hashtags: rareearth, metal, metals, china, export, trade
Video Transcript: It's been a messy few weeks in metals markets, and the latest ructions are playing out in the space for rare earth metals. China announced an export ban on rare earth metals and the magnets that they go into. And this is potentially a really big deal because although rare earth metals are not actually rare, you can find them all over the world and they can be mined in countries all over the place. What is rare is the processing capabilities that are all concentrated primarily in China. What China does is it turns the metals into something that magnet manufacturers can use. And those magnets go into things like wind turbines, electric vehicles, the defense industry, the technology industries. It's really important that countries around the world have secure supply chains of rare earths. And this is exactly the problem at the moment that basically no one does apart from China. How this plays out is a little unclear. They have said that they'll be giving out licenses, so it may not be a total blanket ban. But what is clear is the vulnerability of rare earth supply chains. And this is an issue for countries and companies all over the world. It's not just the US, which is in an escalating trade war with China. It's also countries like Japan and Korea, which import large volumes of rare earths and manufacture magnets. And if countries and companies don't wanna be dependent on one main supplier, China, then they need to build out mines and processing facilities elsewhere. But that can take years. So there's a lot to watch still. For more UN videos visit www.un.org"""

x_finfluencer_examples = """Example Finfluencer Profile 1:
- Profile Image: https://pbs.twimg.com/profile_images/1729294932181921792/lI2kq2Ey_normal.jpg
- Profile Name: Joe Weisenthal
- Profile ID: TheStalwart
- Location: New York City
- Profile Description: One half of Bloomberg's Odd Lots Podcast. One quarter of Light Sweet Crude.
- Profile External Link: https://t.co/tKH5J61Zwy
- Profile Creation Date: 2008-03-07T19:37:09.000000Z
- Verified Profile: False
- Blue Verified Profile: True
- Protected Profile: False
- Number of Followers: 412120 Followers
- Following: 6064 Users
- Total Number of Tweets: 441349
- Number of Favorites: 290679
- Number of Media Content: 32563
- Tweets (Sorted from Newest to Oldest):
Creation Date: 2025-06-13 18:56:32+00:00 
Tweet Text: Container volume at the Port of Los Angeles as 9% lower in May than it was in May 2024 https://t.co/2OwPt9VBQj
Number of Likes: 50
Number of Views: 10189
Number of Retweets: 9
Number of Replies: 6
Number of Quotes: 0
Number of Bookmarks: 7
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-13 18:34:39+00:00 
Tweet Text: Stocks lows of the day. VIX shooting higher https://t.co/uZrI5srlpn
Number of Likes: 42
Number of Views: 12945
Number of Retweets: 8
Number of Replies: 22
Number of Quotes: 0
Number of Bookmarks: 3
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-13 18:27:25+00:00 
Tweet Text: SPX back near its lows of the day https://t.co/JOBAdKPm5x
Number of Likes: 57
Number of Views: 15047
Number of Retweets: 4
Number of Replies: 14
Number of Quotes: 3
Number of Bookmarks: 0
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-13 16:34:16+00:00 
Tweet Text: US-Vietnam trade agreement getting closer. https://t.co/b3nYk8fa7a https://t.co/WXpCfgh3kt
Number of Likes: 66
Number of Views: 19388
Number of Retweets: 14
Number of Replies: 31
Number of Quotes: 2
Number of Bookmarks: 9
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-13 16:01:11+00:00 
Tweet Text: WSJ reports that both Amazon and Walmart have looked into issuing stablecoins. Shares of $V down nearly 5% today. https://t.co/rNBYNphId7 https://t.co/hQoheWlkKJ
Number of Likes: 167
Number of Views: 32720
Number of Retweets: 25
Number of Replies: 27
Number of Quotes: 12
Number of Bookmarks: 39
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-13 14:45:23+00:00 
Tweet Text: It looks like a US-Indonesia trade deal is a done deal https://t.co/qE433h3Izm https://t.co/QUllP8S8ns
Number of Likes: 111
Number of Views: 30480
Number of Retweets: 25
Number of Replies: 16
Number of Quotes: 7
Number of Bookmarks: 24
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-13 14:33:35+00:00 
Tweet Text: RT @cpgrabow: .@TheStalwart: "No one thinks [the Jones Act is] invigorating at all to the US shipbuilding industry, which is almost nonexis‚Ä¶
Number of Likes: 95
Number of Views: 18074
Number of Retweets: 14
Number of Replies: 6
Number of Quotes: 0
Number of Bookmarks: 11
Language: en
Tagged Users: Colin Grabow, Joe Weisenthal
Hashtags: nan


Example Finfluencer Profile 2:
- Profile Image: https://pbs.twimg.com/profile_images/1466544057895641089/FcrS-FrV_normal.jpg
- Profile Name: Liz Ann Sonders
- Profile ID: LizAnnSonders
- Location: Naples, FL 
- Profile Description: Chief Investment Strategist, Charles Schwab & Co., Inc. Disclosures: https://t.co/UPkjXSZ9uc
- Profile External Link: https://t.co/OeUMKHEdFE
- Profile Creation Date: 2015-01-06T23:33:53.000000Z
- Verified Profile: False
- Blue Verified Profile: True
- Protected Profile: False
- Number of Followers: 461967 Followers
- Following: 726 Users
- Total Number of Tweets: 35936
- Number of Favorites: 6524
- Number of Media Content: 25582
- Tweets (Sorted from Newest to Oldest):
Creation Date: 2025-06-13 14:09:05+00:00 
Tweet Text: When asked about how their current household finances compare to 5 years ago, consumers haven't felt this pessimistic since February 2013 per @UMich https://t.co/0YXWdp5HwN
Number of Likes: 244
Number of Views: 25681
Number of Retweets: 61
Number of Replies: 44
Number of Quotes: 13
Number of Bookmarks: 14
Language: en
Tagged Users: University of Michigan
Hashtags: nan

Creation Date: 2025-06-13 14:06:27+00:00 
Tweet Text: June @UMich 1y inflation expectations (blue) down to 5.1% vs. 6.6% prior ‚ 5-10y inflation expectations (orange) down to 4.1% vs. 4.2% prior https://t.co/Pp4cDMsNsK
Number of Likes: 72
Number of Views: 15170
Number of Retweets: 23
Number of Replies: 11
Number of Quotes: 0
Number of Bookmarks: 6
Language: en
Tagged Users: University of Michigan
Hashtags: nan

Creation Date: 2025-06-13 14:02:55+00:00 
Tweet Text: June @UMich Consumer Sentiment Index up to 60.5 vs. 53.6 est. &amp; 52.2 prior expectations up to 58.4; current conditions up to 63.7 https://t.co/4vZcTfa0Sf
Number of Likes: 113
Number of Views: 17760
Number of Retweets: 37
Number of Replies: 10
Number of Quotes: 3
Number of Bookmarks: 4
Language: en
Tagged Users: University of Michigan
Hashtags: nan

Creation Date: 2025-06-13 11:24:49+00:00 
Tweet Text: In 2013, there were 35 major U.S. metro areas in which one could buy a luxury home for less than $1M ‚Ä¶ now, there are only 7 metros 
‚Å¶@Redfin‚Å© https://t.co/FbH8dehySS
Number of Likes: 142
Number of Views: 25205
Number of Retweets: 27
Number of Replies: 26
Number of Quotes: 3
Number of Bookmarks: 10
Language: en
Tagged Users: Redfin
Hashtags: nan

Creation Date: 2025-06-13 11:24:07+00:00 
Tweet Text: Finished goods version of PPI moved up in May to +1.4% year/year vs. +0.4% prior https://t.co/v5LPKXZ4Ni
Number of Likes: 45
Number of Views: 13746
Number of Retweets: 8
Number of Replies: 1
Number of Quotes: 0
Number of Bookmarks: 2
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-13 11:23:25+00:00 
Tweet Text: U.S. household net worth fell by 0.93% in 1Q2025 ‚ largest decline since 3Q2022, but not comparable to that quarter in terms of magnitude https://t.co/p0WZRr7VAd
Number of Likes: 72
Number of Views: 13053
Number of Retweets: 21
Number of Replies: 7
Number of Quotes: 0
Number of Bookmarks: 8
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-13 11:22:24+00:00 
Tweet Text: Big gap between median home list price and median sale price ‚ former up to $426k and latter at $397 per @Redfin https://t.co/votBgSNYUM
Number of Likes: 161
Number of Views: 20367
Number of Retweets: 45
Number of Replies: 9
Number of Quotes: 4
Number of Bookmarks: 29
Language: en
Tagged Users: Redfin
Hashtags: nan

Creation Date: 2025-06-13 11:21:45+00:00 
Tweet Text: AAII members equity allocations ticked up in May, albeit slightly (to 64.3%)
@AAIISentiment https://t.co/R0ZPT8Wptn
Number of Likes: 23
Number of Views: 12744
Number of Retweets: 5
Number of Replies: 4
Number of Quotes: 0
Number of Bookmarks: 4
Language: en
Tagged Users: AAII SentimentSurvey
Hashtags: nan

Creation Date: 2025-06-13 11:21:17+00:00 
Tweet Text: Inflation-adjusted average hourly earnings grew by 1.5% year/year in May slower than prior month but still relatively strong https://t.co/TxAfrOtJlq
Number of Likes: 61
Number of Views: 12819
Number of Retweets: 12
Number of Replies: 4
Number of Quotes: 0
Number of Bookmarks: 3
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-13 11:20:56+00:00 
Tweet Text: Portfolio management component of PPI fell for a second consecutive month in May down by 1% m/m https://t.co/c9zKXsdDjM
Number of Likes: 39
Number of Views: 12745
Number of Retweets: 3
Number of Replies: 5
Number of Quotes: 0
Number of Bookmarks: 2
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-13 11:20:35+00:00 
Tweet Text: Finished consumer goods component in PPI continued to deflate in May down by 6.5% annualized over past three months https://t.co/EQwUO1iA3v
Number of Likes: 55
Number of Views: 12019
Number of Retweets: 11
Number of Replies: 5
Number of Quotes: 0
Number of Bookmarks: 4
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-13 11:20:00+00:00 
Tweet Text: Bull-Bear @AAIISentiment spread ticked up last week, rising to +3.1% as optimism creeps back in https://t.co/R2rc4rPPzy
Number of Likes: 29
Number of Views: 16671
Number of Retweets: 5
Number of Replies: 3
Number of Quotes: 1
Number of Bookmarks: 6
Language: en
Tagged Users: AAII SentimentSurvey
Hashtags: nan

Creation Date: 2025-06-13 11:19:15+00:00 
Tweet Text: 3m annualized change in core PPI eased to +0.85% in May https://t.co/mNkf5aDXql
Number of Likes: 38
Number of Views: 11900
Number of Retweets: 5
Number of Replies: 3
Number of Quotes: 0
Number of Bookmarks: 2
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-12 17:56:53+00:00 
Tweet Text: U.S. household net worth fell by $1.595 trillion in 1Q2025 first decline sine 3Q2023, driven by weakness in equity market https://t.co/i5gGn68V6u
Number of Likes: 187
Number of Views: 16524
Number of Retweets: 40
Number of Replies: 53
Number of Quotes: 2
Number of Bookmarks: 12
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-12 17:02:00+00:00 
Tweet Text: RT @KevRGordon: The Atlanta Fed's Wage Growth Tracker has been at 4.3% for four straight months https://t.co/HFkaBQ8HwF
Number of Likes: 82
Number of Views: 13849
Number of Retweets: 19
Number of Replies: 6
Number of Quotes: 0
Number of Bookmarks: 7
Language: en
Tagged Users: Kevin Gordon
Hashtags: nan


Example Finfluencer Profile 3:
- Profile Image: https://pbs.twimg.com/profile_images/378800000557074315/275249c4889ec856ed268feef4abb00e_normal.jpeg
- Profile Name: Aswath Damodaran
- Profile ID: AswathDamodaran
- Location: New York & San Diego
- Profile Description: Fascinated by finance & markets and like writing about them, but teaching is my passion.
- Profile External Link: https://t.co/2KZsqssYUK
- Profile Creation Date: 2009-04-19T14:54:22.000000Z
- Verified Profile: False
- Blue Verified Profile: True
- Protected Profile: False
- Number of Followers: 428246 Followers
- Following: 3 Users
- Total Number of Tweets: 3022
- Number of Favorites: 89
- Number of Media Content: 976
- Tweets (Sorted from Newest to Oldest):
Creation Date: 2025-06-02 20:15:23+00:00 
Tweet Text: My June 2025 update of the US ERP was disrupted by Moodys' downgrade of the US from Aaa to Aa1, with the potential of altering how I assess the US dollar risk free rate and how I compute equity risk premiums. https://t.co/WUI6H7yz3h
Number of Likes: 364
Number of Views: 57868
Number of Retweets: 54
Number of Replies: 5
Number of Quotes: 5
Number of Bookmarks: 103
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-05-03 16:10:57+00:00 
Tweet Text: Stocks ended April 2025 at about the same levels that they started the month at, but that absence of change in aggregate values masked significant shifts and changes in market sentiment and trust in US institutions. https://t.co/s1DPON1L6N https://t.co/xUlYeGJvJX
Number of Likes: 456
Number of Views: 54692
Number of Retweets: 88
Number of Replies: 13
Number of Quotes: 4
Number of Bookmarks: 96
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-04-25 22:23:06+00:00 
Tweet Text: With three trading days left in the month, I am updating my numbers through April 25. During the last week (4/21-4/25), US stocks were up, risk premiums for equity and bonds were down oil &amp; gold leveled off and bitcoin rose with stocks. https://t.co/ra2jbIIkyU
Number of Likes: 361
Number of Views: 41119
Number of Retweets: 71
Number of Replies: 6
Number of Quotes: 2
Number of Bookmarks: 56
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-04-21 03:00:14+00:00 
Tweet Text: In every market downturn, the advice to buy the dip is widely offered. I look at contrarian investing, in all its different forms, in this post, with the pluses and minuses of each one. https://t.co/Yx5Ku6UcdS
Number of Likes: 1370
Number of Views: 166143
Number of Retweets: 202
Number of Replies: 17
Number of Quotes: 16
Number of Bookmarks: 1059
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-04-19 01:47:05+00:00 
Tweet Text: Week three update of the tariff crisis with a drama about the Fed chair entering the mix. A week of relative stability, with the emphasis on the word "relative". The ERP inched up, treasury rates dropped and equity losses evened out across regions. https://t.co/wy8WGu8ona https://t.co/Dk4j2TJT78
Number of Likes: 383
Number of Views: 46250
Number of Retweets: 56
Number of Replies: 8
Number of Quotes: 3
Number of Bookmarks: 93
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-04-11 23:15:20+00:00 
Tweet Text: Following up on my promise to update equity risk premiums by day, here was what the last week (4/7 - 4/11) delivered. Stocks, T.Bonds and gold were all up (a most unusual combination); the ERP and spreads dipped. The rollercoaster ride continues! https://t.co/jHV9UNujmk https://t.co/JcL5SctZKU
Number of Likes: 478
Number of Views: 58212
Number of Retweets: 86
Number of Replies: 18
Number of Quotes: 1
Number of Bookmarks: 163
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-04-08 13:13:50+00:00 
Tweet Text: The Trump tariffs have roiled markets, with US equity indices down decisively on April 3 and 4. They have the potential to not just reset the pricing of stocks but the economic outlook, near and long term. I don't know what's coming but I try my best here: https://t.co/bwZtRExVb3 https://t.co/rrlWKrCnto
Number of Likes: 820
Number of Views: 139372
Number of Retweets: 124
Number of Replies: 13
Number of Quotes: 8
Number of Bookmarks: 433
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-03-31 00:30:28+00:00 
Tweet Text: Buffeted by talk of tariffs and trade wars on one hand, and by fears of inflation and recession on the other, the S&amp;P 500 buckled in March 2024, down 6.3%; the ERP rose to 4.61% and the expected return to 8.85%.  Spreadsheet: https://t.co/lfZfvQg0tw https://t.co/DtwhmPMQZ7
Number of Likes: 348
Number of Views: 56968
Number of Retweets: 65
Number of Replies: 13
Number of Quotes: 3
Number of Bookmarks: 105
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-03-03 14:33:22+00:00 
Tweet Text: Since 2020, US equity markets have been "resilient" in the face of expert meltdowns. The market got shock treatment in February 2025, and came out bruised, but still standing, with the S&P 500's equity risk premium at 4.35% and the expected return at 8.57%.  https://t.co/ovfiRorvLc
Number of Likes: 435
Number of Views: 58510
Number of Retweets: 61
Number of Replies: 11
Number of Quotes: 5
Number of Bookmarks: 83
Language: en
Tagged Users: nan
Hashtags: nan"""

tiktok_nonfinfluencer_examples = """Example Non-Finfluencer Profile 1
- Profile Image: https://p16-sign-va.tiktokcdn.com/tos-maliva-avt-0068/8caec48ffb58b275f25c86279977bec3~tplv-tiktokx-cropcenter:720:720.jpeg?dr=10399&refresh_token=561d621b&x-expires=1748991600&x-signature=ogwe9zqfJr1TZaxMYxVGAf2V7ZQ%3D&t=4d5b0474&ps=13740610&shp=a5d48078&shcp=f20df69d&idc=no1a
- Profile Name: cbsmornings
- Profile Nickname: CBS Mornings
- Profile Biography: Impactful journalism + exquisite storytelling. 
Watch 7-9 a.m. on CBS 🌞
- Profile Signature: Impactful journalism + exquisite storytelling. 
Watch 7-9 a.m. on CBS 🌞
- Profile Biography Link: https://nevertoolate.cbsnews.com/
- Profile URL: https://www.tiktok.com/@cbsmornings
- Profile Language: en
- Profile Creation Date: 2019-08-26T13:40:05.000Z
- Verified Status: True
- Number of Followers: 3100000.0 Followers
- Following: 34.0 Users
- Total Number of Likes: 246300000.0
- Total Number of Videos: 3794.0
- Total Number of Digg: 0.0
- Private Account: False
- Region: US
- TikTok Seller: False
- Average Engagement Rate: 0.0109051612903225
- Comment Engagement Rate: 0.0002340276497695
- Like Engagement Rate: 0.0106711336405529
- Video Transcripts (Sorted from Newest to Oldest):
Creation Date: 2025-05-31 19:20:01+00:00
Video Description: Rottnest Island off the coast of Australia has become a beloved tourist destination for its adorable resident - Quokkas. The small marsupials are an endangered species, but efforts to conserve them are going well, in part, to how photogenic they are. #quokka #australia #animalsoftiktok #selfie 
Video Duration: 108.0
Number of Likes: 1888.0
Number of Shares: 176.0
View Count: 44000.0
Number of Saves: 80.0
Number of Comments: 23.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.04925
Tagged Users: 
Hashtags: quokka, australia, animalsoftiktok, selfie
Video Transcript: Around 10,000 of these furry creatures live on the tiny island. They're cousins of the kangaroo, but stand only around 20 inches tall. And because they seem to be smiling, quokkas have become a sensation, known as the world's happiest animals. Quokka selfies have gone viral, including snaps by celebrities like Roger Federer, Margot Robbie, and Logan Paul. This way. Smile. And as we found, you don't need to be a wildlife photographer to get up close and personal with a quokka. Thank you. Scientists say quokkas are naturally curious and have adapted to being around tourists. On the Australian mainland, quokkas are threatened by wildfires and feral cats. But on the island, they're almost ridiculously relaxed, even falling asleep outside this cafe on the island's main shopping street. Now, the truth is the quokkas are not actually smiling. That's just a quirk of nature. But they have no natural predators on the island, and that's why they're so fearless and so friendly with humans. Their front teeth are sort of permanently... ..almost permanently exposed, and their lips are just shaped like that. So it looks like they're smiling. Arvid Hogstrom is in charge of conservation on the island. They're just a very chilled animal to start off with. Are they happy? The quokkas on Rottnest are pretty stress-free. They live a pretty easygoing life. The quokkas do lead a privileged existence. It's forbidden to feed or touch them, and they even have right-of-way on the roads. They're a conservation success story, with demand for selfies fuelling the tourism that helps fund their protection.

Creation Date: 2025-05-31 19:10:14+00:00
Video Description: Are dogs better than people? @The Dogist’s Elias Friedman explains the lessons in confidence we all can learn from our four-legged friends. #dogs #pets #dogsoftiktok 
Video Duration: 50.0
Number of Likes: 291.0
Number of Shares: 8.0
View Count: 7666.0
Number of Saves: 12.0
Number of Comments: 11.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.042003652491521
Tagged Users: thedogist
Hashtags: dogs, pets, dogsoftiktok
Video Transcript: dogs better than people? Dogs are better with people, I would say. Dogs possess a lot of the things that we wish we could, like confidence in meeting people, not being self-conscious, not judging people instantly. Just the way a dog walks into a dog park and just smells another dog's butt, you know, we don't really do that. We judge from afar. Which is good for social reasons, but yeah. Yeah, yeah. Dogs do a lot of things that we wish we could do. Just be content with who they are. And hang out while we're sitting here talking and talking, and they're like, ah, whatever. Yeah, they're just going with the flow. They're hilarious. They make us laugh. That's always been the thing for me. I love a good sense of humor, and I feel like dogs have a good sense of humor.

Creation Date: 2025-05-31 18:14:06+00:00
Video Description: Across college campuses, more and more students are using artificial intelligence tools to complete their assignments. Some schools are now using AI detection software to increase academic integrity. However, the programs are not always accurate. #ai #artificalintelligence #chatgpt #college #exams #education #student 
Video Duration: 88.0
Number of Likes: 66300.0
Number of Shares: 3494.0
View Count: 1100000.0
Number of Saves: 4802.0
Number of Comments: 984.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.06870909090909091
Tagged Users: 
Hashtags: ai, artificalintelligence, chatgpt, college, exams, education, student
Video Transcript: get an email three days later saying, Hey, you've been flagged for plagiarism, specifically chachi Bt. And for that you need to contest this, or you take a zero and you fail the class. An AI detection tool incorrectly flagged his work with his scholarship on the line. Rivera reached out to his professor who after a closer look, confirmed he did not cheat. This is known as a false positive. I would have had a very hellish fall semester. We spoke to nearly a dozen students across the country who have been wrongly accused of AI cheating. Dr. So he'll phase he is an associate professor at the University of Maryland. He's published over 100 peer reviewed papers on AI research and AI text detection tools. What false positive rate are you seeing these AI detectors find many of the companies that we have looked into claim their false positive rate is quote unquote, as low as one person. We shouldn't accept detectors with 1% false positive rate because false accusation of AI plagiarism can be quite damaging to the student. These products can return a range of results. I actually wrote a piece of text myself. So I'll just copy the text here. We'll analyze the text. Oops. Oh my gosh. 59% This will be a false positive, I will be accused of AI plagiarism.

Creation Date: 2025-05-31 15:41:32+00:00
Video Description: Two crypto investors have been indicted in the kidnapping and torture of an Italian tourist in a luxury New York City apartment. Prosecutors allege the two men tormented the victim with electric wires, forced him to smoke from a crack pipe and even dangled him over the staircase — all in an attempt to gain his bitcoin password.  #bitcoin #crypto #truecrime #crime #newyorkcity #paris 
Video Duration: 124.0
Number of Likes: 3369.0
Number of Shares: 377.0
View Count: 114200.0
Number of Saves: 183.0
Number of Comments: 157.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.035779334500875656
Tagged Users: 
Hashtags: bitcoin, crypto, truecrime, crime, newyorkcity, paris
Video Transcript: William DiPlessi appeared in a Manhattan courtroom Friday with bags under his eyes, indicted in a Bitcoin kidnapping scheme, along with John Waltz. Prosecutors allege the two men, both cryptocurrency investors in their 30s, lured an Italian millionaire to their ritzy SoHo townhouse, where they held him captive and tortured him over 17 days. The suspects were arrested after the 28-year-old man managed to escape and flag down a traffic cop on the street. Inside the apartment, police say they found body armor, ammunition and a photo of a gun held to the man's head. To be tortured for 17 days in terms of a chainsaw cutting your leg, in terms of putting your feet in water and electrocuting them, in terms of making the person ingest narcotics, horrible crime. Prosecutors say the suspects took the alleged victim's electronics and threatened to kill his family unless he gave up the password to his Bitcoin wallet worth millions. As the NYPD parsed through evidence in SoHo this week, French police have arrested more than 20 people for a number of kidnapping plots targeting crypto tycoons, which includes the attempted abduction of an investor's daughter by masked men on a busy Paris street earlier this month. I don't go to certain crypto meetings anymore where I had the habit to go before. I try to park as close as possible to the destination when I have to go somewhere. The string of attacks, both abroad and now domestic, have some Bitcoin bigwigs worried for their safety. Bitcoin's worth over $100,000 and people are still posting screenshots of their holdings, travel plans of where they're going to be. So it's really an attacker's dream because they don't really need to do much, too much research out there to target these people that now have quite a lot of value in their crypto holdings. And so, unfortunately, these attacks are, again, becoming more gruesome, more brazen and are increasing in likelihood and severity.

Creation Date: 2025-05-31 12:58:00+00:00
Video Description: Rough weather and tough terrain are posing a challenge for investigators searching for Grant Hardin, a former police chief and convicted killer who escaped from prison.  Ian Lee has the latest on intense manhunt for the man known as the  "Devil in the Ozarks." #news #ozarks #arkansas #manhunt #news #crime #crimetok 
Video Duration: 96.0
Number of Likes: 25900.0
Number of Shares: 1315.0
View Count: 576500.0
Number of Saves: 1338.0
Number of Comments: 767.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.050858629661751954
Tagged Users: 
Hashtags: news, ozarks, arkansas, manhunt, news, crime, crimetok
Video Transcript: Investigators are nowhere closer to finding escaped convicted killer and rapist Grant Hardin. Rough weather and tough terrain hamper their search. Whether it's trees, whether it's branches, whether it's abandoned sheds, there's just so many places to hide. From the air, a canopy of leaves provides some cover from drones and helicopters, which is making an already difficult search more challenging for law enforcement. How confident are you that he is still in northern Arkansas? We still feel fairly confident. I mean, the search continues primarily in the north central Arkansas region, mainly for the fact that we have had nothing verifiable to put him outside of this area. Currently, investigators are looking into whether Hardin's job in the prison kitchen helped him get the materials to make his fake law enforcement uniform that he used in his escape. Dogs initially picked up his scent, tracking it west of the prison, but they quickly lost it after a downpour of rain. How is he able to survive in these woods? I mean, people in northern Arkansas, they're resourceful. We have to assume that he has some sort of survival knowledge. While the search for Hardin moves on, it's not stopping campers from moving in. Is there any concern about him being in these woods, potentially? I mean, naturally there has to be some concern, right? Camper Luke Henson and his family say they're taking precautions. Most people have guns, so those are options. Hopefully it doesn't come to that point, but if it does come to that point, we have some defense mechanism.

Creation Date: 2025-05-30 16:58:42+00:00
Video Description: Eddie Murphy’s son Eric and Martin Lawrence’s daughter Jasmin eloped earlier this month, Murphy revealed to Jennifer Hudson on Thursday. Murphy joked that Lawrence, now his in-law, won’t “have to pay for the big wedding.” #eddiemurphy #martinlawrence 
Video Duration: 23.0
Number of Likes: 3427.0
Number of Shares: 116.0
View Count: 85100.0
Number of Saves: 166.0
Number of Comments: 28.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.04391304347826087
Tagged Users: 
Hashtags: eddiemurphy, martinlawrence
Video Transcript: And two of comedy's biggest names can now call themselves in-laws. Eddie Murphy's son, Eric, and Martin Lawrence's daughter, Jasmine, eloped earlier this month. Murphy told Jennifer Hudson yesterday that he was disappointed it wasn't a big bash for one specific reason. The couple had a small ceremony, just the two of them and the preacher. Murphy says he thinks they'll still have a big party, so Lawrence may not be off the hook yet.

Creation Date: 2025-05-30 16:05:24+00:00
Video Description: Haribo is recalling packs of sweets in the Netherlands after some were found to contain traces of cannabis. Local Dutch media reported that a family became "quite ill" after eating the candy and reported the incident to police. #candy #haribo #netherlands 
Video Duration: 28.0
Number of Likes: 578900.0
Number of Shares: 179900.0
View Count: 8400000.0
Number of Saves: 42129.0
Number of Comments: 10300.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.09657488095238095
Tagged Users: 
Hashtags: candy, haribo, netherlands
Video Transcript: Candy company Haribo has issued a warning that some of its Happy Cola candy, sold in the Netherlands, reportedly tested positive for cannabis. Local media reports that several people, including at least one family, became quote, quite ill. Oh no! It's like couch luck to me after eating the candy, prompting a police investigation. It is not clear if cannabis was in this drug, or in this candy, how that drug made its way into the treats.

Creation Date: 2025-05-30 15:32:17+00:00
Video Description: Your March Madness bracket could look a little different next year. NCAA president Charlie Baker is floating the idea of adding up to eight additional teams to the men’s tournament next year. #ncaa #marchmadness #basketball 
Video Duration: 33.0
Number of Likes: 560.0
Number of Shares: 29.0
View Count: 28600.0
Number of Saves: 17.0
Number of Comments: 24.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.022027972027972027
Tagged Users: 
Hashtags: ncaa, marchmadness, basketball
Video Transcript: Your March Madness bracket might look a little different next year. This is big. Yesterday, the NCAA president, Charlie Baker, said he sees value in expanding the college basketball tournament by a handful of teams and that he wants to reach a decision in the next few months. So, Baker floated the idea of expanding from 68 teams to 72 teams or even 76. He said the current formula has flaws and that it would be beneficial to give more opportunities to teams. The flaws are there's money on the table that we can put our crawls around. Let's just make it February and March Madness.

Creation Date: 2025-05-30 14:50:17+00:00
Video Description: Video shows thieves crawling on the ground of a Los Angeles coffee shop before spray painting the security camera. They then cut a hole in the wall to access a safe in the jewelry store next door. There, the thieves stole $2 million in cash and jewelry. No arrests have been made. #losangeles #jewelry 
Video Duration: 34.0
Number of Likes: 292.0
Number of Shares: 24.0
View Count: 17400.0
Number of Saves: 19.0
Number of Comments: 9.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.019770114942528734
Tagged Users: 
Hashtags: losangeles, jewelry
Video Transcript: Thieves hit a jewelry store near Los Angeles, and they found a sneaky way to avoid the motion detectors. That's someone on the floor, army-crawling along a coffee shop next door, then spray-painting the camera. The robbery began when they broke through the roof to get into that store. They then cut a hole in the wall right into the jewelry store's safe. Wow. Inside that safe, and they made off with this, $2 million. Whoa. In cash and jewelry. Yeah, those are empty boxes. No arrests have been made in this case.

Creation Date: 2025-05-30 14:44:47+00:00
Video Description: Broadway legend Audra McDonald, who became the most Tony Award-nominated performer of all time this year, tells Gayle King about the physical toll of performing eight “Gypsy” shows a week: “It’s finding those quiet moments where I can to just sort of let her go.” Tune in Tuesday for more from their conversation. #broadway #audramcdonald 
Video Duration: 42.0
Number of Likes: 3067.0
Number of Shares: 26.0
View Count: 43800.0
Number of Saves: 162.0
Number of Comments: 22.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.07481735159817351
Tagged Users: 
Hashtags: broadway, audramcdonald
Video Transcript: I've heard you described as an emotional athlete. Christine Baranski said she's a Navy SEAL. To do that night after night after night, what is the physical toll, if any, it takes on you night after night? The physical toll is huge, I will say that. And I don't know what I'm doing to get through that part, I have to say, in terms of how I take care of myself. I'm a mom, I still have an eight-year-old at home. I'm a wife, I wanna be present in my home life. So it's finding those quiet moments where I can to just sort of let her go. One of the things I do is try and leave Rose at the theater. That's one of the things that's really important to me, to leave her there, not take her home with me.


Example Non-Finfluencer Profile 2
- Profile Image: https://p16-sign-va.tiktokcdn.com/tos-maliva-avt-0068/7311513410821160966~tplv-tiktokx-cropcenter:720:720.jpeg?dr=10399&refresh_token=ffbb73fa&x-expires=1748991600&x-signature=fFLCTPguMBby8PgH7Gc%2B2XYa%2Bt4%3D&t=4d5b0474&ps=13740610&shp=a5d48078&shcp=81f88b70&idc=no1a
- Profile Name: grahamstephan
- Profile Nickname: Graham Stephan
- Profile Biography: 4.9 Million Subscribers on YouTube
Host of The Iced Coffee Hour Podcast ☕️
- Profile Signature: 4.9 Million Subscribers on YouTube
Host of The Iced Coffee Hour Podcast ☕️
- Profile Biography Link: http://www.youtube.com/c/GrahamStephan
- Profile URL: https://www.tiktok.com/@grahamstephan
- Profile Language: en
- Profile Creation Date: 2019-01-07T13:46:20.000Z
- Verified Status: True
- Number of Followers: 950100.0 Followers
- Following: 52.0 Users
- Total Number of Likes: 40900000.0
- Total Number of Videos: 1039.0
- Total Number of Digg: 0.0
- Private Account: False
- Region: US
- TikTok Seller: False
- Average Engagement Rate: 0.1696383206199901
- Comment Engagement Rate: 0.0010310826007234
- Like Engagement Rate: 0.1686072380192666
- Video Transcripts (Sorted from Newest to Oldest):
Creation Date: 2025-06-01 23:33:58+00:00
Video Description: No Tax on Tips! #taxtips #tippingculture #servertok 
Video Duration: 46.0
Number of Likes: 183.0
Number of Shares: nan
View Count: 4638.0
Number of Saves: 1.0
Number of Comments: 16.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.043122035360069
Tagged Users: 
Hashtags: taxtips, tippingculture, servertok
Video Transcript: a brand new tax plan, and that would be no tax on tips. In order to actually qualify, you must work in an industry where tipping is customary, like a server, bartender, hairstylist, or taxi driver. Two, tips must be paid voluntarily and not negotiated or included automatically as a service charge. Three, you have to have an income below $150,000 a year. Four, the maximum you could claim is $25,000. So anything beyond this is not tax free. I'm going to be completely honest. I think it's somewhat worthless because, let's be real, most people getting cash tips are not reporting them anyway. In fact, some people might not even want to report cash tips, even if it is tax free, because it might disqualify them from receiving other government subsidies that only people with lower income qualify for.

Creation Date: 2025-06-01 06:49:00+00:00
Video Description: Trump Is Paying $1,000 To Become a Parent! #irs #taxtips #taxrefund 
Video Duration: 46.0
Number of Likes: 378.0
Number of Shares: 20.0
View Count: 6942.0
Number of Saves: 25.0
Number of Comments: 18.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.06352636127917027
Tagged Users: 
Hashtags: irs, taxtips, taxrefund
Video Transcript: If you have a child, you could receive $8,000 within a Trump account. Eligible children born 2025 through 2028 in the United States will receive a one-time tax credit of $1,000 funded by the IRS into an account that could then be invested. The goal here is that with compound interest, $1,000 will grow to an amount that could then be used towards qualifying expenses like higher education, business startup costs, or a first-time home purchase. Of course, non-qualifying expenses, like Coachella, would be subject to a 10% penalty plus ordinary income taxes. And if you don't use any of the money by the age of 31, the account will automatically be cashed out. In terms of how much this could grow to, $1,000 invested at birth at an average return of 9%, that would equate to more than $13,000.

Creation Date: 2025-05-30 19:10:10+00:00
Video Description: I Bought a Tesla Without Test Driving It First #teslamodel3 #teslatok #carreview 
Video Duration: 56.0
Number of Likes: 919.0
Number of Shares: 2.0
View Count: 20000.0
Number of Saves: 27.0
Number of Comments: 33.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.04905
Tagged Users: 
Hashtags: teslamodel3, teslatok, carreview
Video Transcript: I just bought the $35,000 Tesla Model 3 without ever having seen it and without ever having driven one before. The standard Model 3 begins at the price of $35,000. If you increase your budget just $2,000, you'll get an extra 20 miles of range, a slightly higher top speed, power adjusted seats, premium interior, and upgraded audio. If you get any other color besides black, it'll cost you up to $2,500 more if you want red. So we're going with black today. Next they give you the option to upgrade to 19-inch wheels for an extra $1,500. That's not going to happen. Now we have the choice of interior color, either black or white, but white is an extra $1,000 and I would be worried about getting that dirty, so black it is. They also offer full self-driving for an extra $5,000. This includes summoning your car, but for $5,000, I think I could just do that myself. This is the 2019 Tesla Model 3.

Creation Date: 2025-05-30 00:51:51+00:00
Video Description: Touring a $25 MILLION Mansion #mansiontour #luxuryhomes #luxuryrealestate 
Video Duration: 50.0
Number of Likes: 442.0
Number of Shares: 4.0
View Count: 9463.0
Number of Saves: 16.0
Number of Comments: 26.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.05156926978759379
Tagged Users: 
Hashtags: mansiontour, luxuryhomes, luxuryrealestate
Video Transcript: a $25 million mansion. We're gonna go there right now and check out just how crazy this place really is. So how many bedrooms, how many bathrooms? Even though it's a very large home, we have three bedrooms, we have 12 bathrooms. And we have a same projection system you'd find in a high-end commercial theater. How much does this cost to build? We're about a half million dollars all in on this theater. Wow. We also have a super toilet. What is a super toilet? Lid automatically opens when you walk nearby. This is the future. It is a heated toilet, so it's never cold. Oh, good, good. I was worried for a second it wasn't gonna be heated. This is the car elevator. I don't think many people have seen an elevator for a car. The owner can pull into this car elevator and elevate up to the second floor and walk right into his master bedroom. Everyone needs one of those.

Creation Date: 2025-05-27 22:13:03+00:00
Video Description: Should You Pay Off Debt Or Invest First? #investing #financialadvicefriday #businesstips 
Video Duration: 26.0
Number of Likes: 674.0
Number of Shares: 4.0
View Count: 14700.0
Number of Saves: 28.0
Number of Comments: 17.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.049183673469387755
Tagged Users: 
Hashtags: investing, financialadvicefriday, businesstips
Video Transcript: We recently gave someone £5,000 to start their business, and they used it to pay off their debt, and now they have no money to start their business, so... That's a conundrum. I've always liked seeing if you could make any amount of money without taking on any debt, and if that works, then you could replicate the process. Hopefully they'd be able to go and get a loan for their business at a lower interest rate than the debt they paid off. I probably would have paid half the debt off, the high-interest one, and then used the rest to grow your business so you can make income to pay off the rest of the debt.

Creation Date: 2025-05-26 18:01:00+00:00
Video Description: This Disney Actor Lost ALL Her Money #childactor #disneykids #actorslife 
Video Duration: 59.0
Number of Likes: 963.0
Number of Shares: 3.0
View Count: 37700.0
Number of Saves: 48.0
Number of Comments: 18.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.02737400530503979
Tagged Users: 
Hashtags: childactor, disneykids, actorslife
Video Transcript: Christy Romano, she was on even Stevens, and she's also the voice on Kim Possible. And this is her video of how I lost all my money. I started making money with Disney when I was 16, and there's a law called the Coogan Law that protects my from their parents spending all their money. I was never told how much money I was making. I decided to part ways with my family because I didn't like the way that my money was being managed. The Coogan Law basically just says that if a child is working, the parents must set aside 15% of their money that can't be touched whatsoever until they turn 18 years old. And this is basically enacted so that the parents don't just receive all the money the kid makes, spend it, and then the child turns 18 and has no more money left. Because believe it or not, that was a common practice where parents would just siphon off the money from the child, the child would support the entire family, and then they have nothing left. At least 15% goes into the child's name, but 15% in hindsight is really not that much.

Creation Date: 2025-05-25 18:17:00+00:00
Video Description: Why I LOVE DEBT #businesstips #roi #financialadvice 
Video Duration: 34.0
Number of Likes: 636.0
Number of Shares: 5.0
View Count: 11100.0
Number of Saves: 50.0
Number of Comments: 16.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.0636936936936937
Tagged Users: 
Hashtags: businesstips, roi, financialadvice
Video Transcript: lot of people talk about debt as a good thing. How do you know if it's good or bad debt? I have a lot of good debt and I view any good debt as something that makes you more money than you spend. Let's say you have a business and you're able to borrow money at 10% but that borrowed money makes you a 50% ROI because you could run ads with it. Then it's fantastic debt or a low interest mortgage. Let's just say it's 3% but inflation is 3%. That's good debt. If you have a car and you're paying 6% but that car is used for business that makes you 10%. I think that's good. So if it makes you more money than you spend, it's not the end of the world to keep it.

Creation Date: 2025-05-24 13:30:18+00:00
Video Description: A 401(k) Employer Match Is LITERALLY FREE MONEY #401k #retirementplanning #retirementsavings 
Video Duration: 22.0
Number of Likes: 2572.0
Number of Shares: 37.0
View Count: 112000.0
Number of Saves: 197.0
Number of Comments: 52.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.025517857142857144
Tagged Users: 
Hashtags: 401k, retirementplanning, retirementsavings
Video Transcript: If you work in a place that offers a 401k contribution, it's free money. Some employers will offer what's called a 401k match, which is basically they say, hey, if you contribute a dollar we're gonna match your contribution. It's a guaranteed 100% return immediately. It doesn't exist anywhere else other than a employer match on a 401k. So if you had that offer, I would be contributing up to the maximum. It's literally free money.

Creation Date: 2025-05-22 19:26:00+00:00
Video Description: Truck Drivers Make $100,000 a Year! #cdl #cdllife #truckdriver 
Video Duration: 32.0
Number of Likes: 1221.0
Number of Shares: 10.0
View Count: 21600.0
Number of Saves: 41.0
Number of Comments: 34.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.06046296296296296
Tagged Users: 
Hashtags: cdl, cdllife, truckdriver
Video Transcript: I'm a female truck driver and I make over six figures a year. Last year was able to make $144,208. My truck lease payment, $49,200. My truck maintenance, $2,665. My fuel cost, $19,336. I was able to profit $71,309. So she owns her own truck. So she's not just like driving someone else and she maintains her own and she's able to keep the profits. Wow.

Creation Date: 2025-05-21 19:37:24+00:00
Video Description: The Power Of Your First $10,000 #savingmoney #moneysavingtips #savings 
Video Duration: 47.0
Number of Likes: 1290.0
Number of Shares: 23.0
View Count: 19800.0
Number of Saves: 90.0
Number of Comments: 19.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.07181818181818182
Tagged Users: 
Hashtags: savingmoney, moneysavingtips, savings
Video Transcript: This is why your first $10,000 is so significant. With this amount of money, you'll gain enough financial confidence to know that if you could do this, you could level up in practically any area of your life that you put your mind to. Just the idea that I could go and buy a used car outright gave me such a sense of accomplishment. $10,000 is an amount of money where you'd be able to cover almost any immediate financial emergency without going into debt. For example, if your car breaks down, most repairs are probably going to be a few thousand dollars. But if your pet gets sick and you get hit with a vet bill, you've got it covered. If you lose your job, you've got $10,000 sitting there until you find a replacement. This just takes the small stresses out of life. It starts opening up the door to take on a little bit more risk, and it gives you the autonomy to make decisions based on choice and not necessity.


Example Non-Finfluencer Profile 3
- Profile Image: https://p16-sign-va.tiktokcdn.com/tos-maliva-avt-0068/d8b707e8ab71dc4c0adcf27e4c841ade~tplv-tiktokx-cropcenter:720:720.jpeg?dr=10399&refresh_token=67761e5f&x-expires=1748991600&x-signature=O3URbM2AOxQ3qamhgkEAOG87RKE%3D&t=4d5b0474&ps=13740610&shp=a5d48078&shcp=81f88b70&idc=no1a
- Profile Name: bellapoarch
- Profile Nickname: Bella Poarch
- Profile Biography: nan
- Profile Signature: nan
- Profile Biography Link: https://bellapoarch.lnk.to/WYALHVideo
- Profile URL: https://www.tiktok.com/@bellapoarch
- Profile Language: en
- Profile Creation Date: 2019-10-29T19:33:46.000Z
- Verified Status: True
- Number of Followers: 93900000.0 Followers
- Following: 649.0 Users
- Total Number of Likes: 2400000000.0
- Total Number of Videos: 787.0
- Total Number of Digg: 0.0
- Private Account: False
- Region: US
- TikTok Seller: False
- Average Engagement Rate: 0.0047991315989654
- Comment Engagement Rate: 3.874821238399513e-05
- Like Engagement Rate: 0.0047603833865814
- Video Transcripts (Sorted from Newest to Oldest):
Creation Date: 2025-05-03 00:59:27+00:00
Video Description: #philippines 🇵🇭
Video Duration: 62.0
Number of Likes: 695400.0
Number of Shares: 6119.0
View Count: 10900000.0
Number of Saves: 20583.0
Number of Comments: 5130.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.06671853211009174
Tagged Users: 
Hashtags: philippines
Video Transcript: I flew to the Philippines and got eye surgery. I'm going to the Philippines and my eye is swollen, so let's go. I don't know why, but every single time I travel, shit always happens. It's been 14 years since I've been back to the Philippines, and there's no way I'm gonna cancel over a sty. After my 14-hour flight, we headed straight to the hospital. Look, a jeepney. Wait, look. It's a jeepney. I had an important shoot in a week, so I decided to get last-minute surgery. The surgery went so smooth, and I'm so grateful for the Filipino nurses and doctors that took care of me. I did a fitting after my surgery. Yes, a fitting, because we gotta look cute. And then headed to the airport after. I could literally feel my eye swelling. When we got to our hotel in Bohol, we were greeted with a surprise. I literally didn't realize how crazy I looked. And now for the fun part. Now let's unbox. I've literally been traveling for 30 hours. I look insane. I took the bandage off, and OMG. No bruise, no swelling. Purr. And now I'm ready for Philippines. Whee!

Creation Date: 2025-04-28 20:19:44+00:00
Video Description: More omnichord content on IG reels🤍 follow me there !!
Video Duration: 15.0
Number of Likes: 160100.0
Number of Shares: 2153.0
View Count: 4600000.0
Number of Saves: 11542.0
Number of Comments: 2737.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.038376521739130434
Tagged Users: 
Hashtags: 
Video Transcript: Such as you, me, and the wreckage of the world Must be so confusing for a little girl

Creation Date: 2025-04-27 17:50:49+00:00
Video Description: My cat is a star🐱new reel on IG. link in bio before he unfollows me !!
Video Duration: 5.0
Number of Likes: 636000.0
Number of Shares: 8530.0
View Count: 8100000.0
Number of Saves: 23855.0
Number of Comments: 3895.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.08299753086419753
Tagged Users: 
Hashtags: 
Video Transcript: But now I'm trophied up, I'm sayin' like Boom, bop, boom, boom, boom, bop

Creation Date: 2025-04-08 20:36:22+00:00
Video Description: Let’s go to the studio with my @Vans Super Lowpro💗 #vanspartner 
Video Duration: 19.0
Number of Likes: 552700.0
Number of Shares: 2750.0
View Count: 19700000.0
Number of Saves: 17307.0
Number of Comments: 3324.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.029242690355329948
Tagged Users: vans
Hashtags: vanspartner
Video Transcript: Get ready with me to have a chill day at the studio Let's go! And we are styling the super low pro A cutie! Red or green? Pick the red! Pick the red! Green Sweet, I don't know Yo, check this out Yo, how did you do that? I'm ready Now let's go write some songs

Creation Date: 2025-03-22 23:49:01+00:00
Video Description: Will You Always Love Her? OUT NOW❤️‍🩹
Video Duration: 19.0
Number of Likes: 106800.0
Number of Shares: 920.0
View Count: 4800000.0
Number of Saves: 5114.0
Number of Comments: 2024.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.02392875
Tagged Users: 
Hashtags: 
Video Transcript: Will you always love her? Her smile and her hair say that you don't care But your eyes don't lie, you can't help but stare She looks nothing like me, no she's nothing like me

Creation Date: 2025-03-14 01:00:11+00:00
Video Description: Englishera halata🇵🇭
Video Duration: 15.0
Number of Likes: 1600000.0
Number of Shares: 68600.0
View Count: 14800000.0
Number of Saves: 100075.0
Number of Comments: 11500.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.12028209459459459
Tagged Users: 
Hashtags: 
Video Transcript: Yeah, I heard about that. It's crazy. Also, girl, I'm starving. Honestly, me too. I'm kinda hungry. I know there's like a lot of nearby restaurants. Restaurant, kono. Ay, Jollibee! Oh my God! Where? Over there! Ano ba yan? English Era, halata.

Creation Date: 2025-03-13 00:18:00+00:00
Video Description: Lmfao
Video Duration: 9.0
Number of Likes: 1200000.0
Number of Shares: 13100.0
View Count: 12700000.0
Number of Saves: 40226.0
Number of Comments: 8042.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.09932031496062992
Tagged Users: 
Hashtags: 
Video Transcript: That's a bad bitch for you Middle finger to my haters Are you bitch jealous? Fuck y'all Been knew why was that bitch I didn't need a man to tell me The fuck?

Creation Date: 2025-03-06 19:22:52+00:00
Video Description: Listening to this song is self care🎧
Video Duration: 16.0
Number of Likes: 74000.0
Number of Shares: 510.0
View Count: 3000000.0
Number of Saves: 3915.0
Number of Comments: 1981.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.026802
Tagged Users: 
Hashtags: 
Video Transcript: So, do I keep on wasting time? Leaves are changing color Dark blue in the summer Guess I'll always wonder Will you always love her?

Creation Date: 2025-03-05 00:19:51+00:00
Video Description: GRWM to run errands in my Premium Old Skools @Vans🤍
Video Duration: 29.0
Number of Likes: 176800.0
Number of Shares: 894.0
View Count: 6100000.0
Number of Saves: 6806.0
Number of Comments: 1903.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.030557868852459015
Tagged Users: vans
Hashtags: 
Video Transcript: Get ready with me to run my errands. Good morning. Okay, let's get ready. Did my makeup and hair. Much better. Okay, for the fun part. Let's pick an outfit. First things first, let's pick which Vans. Eenie, meenie, miney, moe. Nice premium old school. Nice. Ooh, yeah. I was thinking it would look cute with this romper. This or this? Blue. Pick the blue. This. And now I'm ready to do my errands. Get it. Let's get it. Let's see our list for today. We got eggs, dish washing soap, cat litter.

Creation Date: 2025-02-20 21:01:37+00:00
Video Description: Lol 
Video Duration: 14.0
Number of Likes: 59400.0
Number of Shares: 746.0
View Count: 3400000.0
Number of Saves: 3719.0
Number of Comments: 2357.0
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): 0.019477058823529413
Tagged Users: 
Hashtags: 
Video Transcript: And if you really want to get it back, go ahead Staying here together might as well be the end So, do I keep on wasting time?"""

x_nonfinfluencer_examples = """Example Non-Finfluencer Profile 1
- Profile Image: https://pbs.twimg.com/profile_images/1803039284108083200/i0blVR4I_normal.jpg
- Profile Name: Aleyda Solis 🕊️
- Profile ID: aleyda
- Location: Remote / Spain
- Profile Description: SEO Consultant, Speaker & Author. @Orainti Founder @Remotersnet Co-Founder @CrawlingMondays Host #SEOFOMO + #MarketingFOMO + https://t.co/yvg9bT4orr Maker @mujeresEnSEO
- Profile External Link: https://t.co/invAGT4oSH
- Profile Creation Date: 2007-01-15T14:33:19.000000Z
- Verified Profile: False
- Blue Verified Profile: True
- Protected Profile: False
- Number of Followers: 168428 Followers
- Following: 932 Users
- Total Number of Tweets: 131358
- Number of Favorites: 4966
- Number of Media Content: 14360
- Tweets (Sorted from Newest to Oldest):
Creation Date: 2025-06-19 19:29:27+00:00 
Tweet Text: 🚨 New Post: I compare top traffic AI Search prompts vs Traditional search queries for the same pages across different pages types: Here are the key differences and the implications when optimizing your content 👇
https://t.co/WXJporXMo4 https://t.co/filOS0nqxY
Number of Likes: 43
Number of Views: 3500
Number of Retweets: 15
Number of Replies: 2
Number of Quotes: 0
Number of Bookmarks: 42
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-18 08:36:18+00:00 
Tweet Text: Useful "ChatGPT search query extractor" bookmarklet to easily obtain search queries from any ChatGPT conversation that grounds its response using a web search with Bing by @ziggyshtrosberg  - Check it out:  https://t.co/97kEx7g4P3 https://t.co/NQIQzaJJ8K
Number of Likes: 70
Number of Views: 5407
Number of Retweets: 13
Number of Replies: 3
Number of Quotes: 1
Number of Bookmarks: 83
Language: en
Tagged Users: Ziggy Shtrosberg
Hashtags: nan

Creation Date: 2025-06-17 07:04:46+00:00 
Tweet Text: 🚨 AI Mode data is now officially in Google Search Console: Google has added an AI Mode section to their Google Search Console documentation, describing how the data is shown 👇
* Clicking a link to an external page in AI Mode counts as a click.
* Standard impression rules apply.
* Position in AI Mode follows the same methodology as a Google Search results page. Generally, carousel and image blocks within AI Mode are calculated using the standard position rules for those elements.
* If a user asks a follow-up question within AI Mode, they are essentially performing a new query. All impression, position, and click data in the new response are counted as coming from this new user query.
Check it here: https://t.co/ZLnfSn90go
/HT @MrDannyGoodwin @glenngabe
Number of Likes: 98
Number of Views: 7854
Number of Retweets: 32
Number of Replies: 1
Number of Quotes: 2
Number of Bookmarks: 60
Language: en
Tagged Users: Danny Goodwin, Glenn Gabe
Hashtags: nan

Creation Date: 2025-06-16 22:07:26+00:00 
Tweet Text: Learn about the Impact of AI Search on Web Publishers in today's #SEOFOMO TL;DR with special guest Barry Adams 📰, covering:
* Google AI Mode: First Thoughts & Survival Strategies by Barry Adams
* How Publishers Can Survive (and Thrive) in the Age of AI Search with @lilyraynyc and @gfiorelli1 
* Have you ever wondered why Google sometimes shows the chunk used for a synthetic answer in AI Overviews / AI Mode, and sometimes not? by @gfiorelli1 
* Writing and optimizing content for NLP-Driven SEO by @jbobbink - Freelance SEO Consultant
* 86% of Top Mentioned Sources Are Not Shared Across ChatGPT, Perplexity, and AI Overviews by @patrickstox 
* Free Looker Studio template to analyze traffic from AI chats by @IvanPalii 
Sponsored by @Similarweb  🙌
Watch the full episode here: https://t.co/IMYn9uoBPO
Number of Likes: 41
Number of Views: 3979
Number of Retweets: 12
Number of Replies: 2
Number of Quotes: 0
Number of Bookmarks: 31
Language: en
Tagged Users: Jan-Willem Bobbink, Gianluca Fiorelli, Patrick Stox, Similarweb, Ivan Palii 🇺🇦, Lily Ray 😏
Hashtags: SEOFOMO

Creation Date: 2025-06-16 18:09:56+00:00 
Tweet Text: The AI Search Content Optimization Checklist [With Google Sheets to copy/paste] 👇
I’ve created an AI Search Content Optimization Checklist, going through the most important aspects to take into account to optimize your content for AI search answers along with their importance and how to take action, going through: 
1. Optimize for Chunk-Level Retrieval
2. Optimize for Answer Synthesis
3. Optimize for Citation-Worthiness
4. Optimize for Topical Breadth and Depth
5. Optimize for Multi-Modal Support
6. Optimize for Content Authoritativeness Signals
7. Optimize for Personalization Resilient Content
8. Optimize for content crawlability and indexability
Check it out and access the Google Sheets here:
https://t.co/e87HMPhDce
Number of Likes: 262
Number of Views: 17327
Number of Retweets: 76
Number of Replies: 10
Number of Quotes: 1
Number of Bookmarks: 412
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-15 18:18:00+00:00 
Tweet Text: The Latest in SEO + AI Search? Here are the top news and resources from today's hashtag#SEOFOMO 👇
* ​Pichai: Google AI Mode Will Be Incorporated Into The Main Search by @rustybrick 
* ​How Publishers Can Survive (and Thrive) in the Age of AI Search with @lilyraynyc  & @gfiorelli1 
* ​Free Looker Studio template to analyze traffic from AI chats by @IvanPalii 
* ​3 examples of product-led SEO by @Kevin_Indig 
* ​Why user-generated content works well for SEO by @torylynne 
* ​Old Hat SEO is Finally Dead. The Bar Has Been Risen For The Next Generation. by @NickLeRoy 
* Optimize for AI Search (GEO, AEO, LLMO) - New Section in #LearningSEO
* ​Google AI Mode: First Thoughts & Survival Strategies by @rustybrick 
* ​Chunked, Retrieved, Synthesized - Not Crawled, Indexed, Ranked by @DuaneForrester 
* ​How Content Structure Matters for AI Search by @chrisgreenseo 
* Much more! Including SEO Jobs, events, tools... 
Read it here (and subscribe to avoid missing out):
https://t.co/JOtkvq42B6
Number of Likes: 43
Number of Views: 5362
Number of Retweets: 15
Number of Replies: 3
Number of Quotes: 2
Number of Bookmarks: 27
Language: en
Tagged Users: duane forrester, Gianluca Fiorelli, Nick LeRoy, Tory (Lynne) Gray, Kevin_Indig, Ivan Palii 🇺🇦, Barry Schwartz, Lily Ray 😏, Chris Green
Hashtags: LearningSEO

Creation Date: 2025-06-13 08:33:20+00:00 
Tweet Text: Announcing the #SEOFOMO x @semrush Barcelona Free Meetup 👇
🗓️ September 4, 2025 - From 6pm to 9pm
📍 Aticco Bogatell (Coworking Poblenou), Barcelona
🍹 Food, Drinks, Quizzes, Giveaways & lots of Fun Networking
🤖 Panels about SEO Trends & AI Search in both English and Spanish 
🎙️ With experienced Spanish and International SEO specialists @SEOJoBlogs, @mjcachon, @ikhuerta, @ghostmou, @ClaraSoteras, Inma Cruz Fuentes, @silvia_smp, @gfiorelli1, @gemmafontane, @TaylorDanRW, @WillKennard, @yagmrsmsk, @NikkiRHalliwell, @Giridja, @JudithLewis and more. 
💰 Completely free! 
Where and how to register? Free registration will open on Sunday June 15 with a link to the registration page to be included in the next #SEOFOMO newsletter and later shared over here on social media. 
Registration is on a first-come, first-served basis, so if you haven't yet, register to the #SEOFOMO newsletter to avoid missing a spot: https://t.co/Dxleipf8tb
I want to thank @semrush  and in particular @SEOJoBlogs and yagmrsmsk for their support 🙌
See you in Barcelooooona 💪
Number of Likes: 38
Number of Views: 4125
Number of Retweets: 12
Number of Replies: 2
Number of Quotes: 5
Number of Bookmarks: 4
Language: en
Tagged Users: Dan Taylor, Gemma Fontané, Judith Lewis, Gianluca Fiorelli, Yagmur Simsek, Iñaki Huerta 🦑/📊/🔝, Clara Soteras, Semrush, MJ Cachón, J Turnbull 🇺🇦, Will Kennard, Alfonso Moure, Nikki Halliwell, 🐝 Olesia Korobka 💙💛🐝, Silvia Martin
Hashtags: SEOFOMO

Creation Date: 2025-06-13 07:04:38+00:00 
Tweet Text: How we’re adapting SEO for LLMs and AI search by Kevin Corbett & @cramforce  from @vercel :
"Search isn’t just about ranking anymore. It’s about being surfaced in new places, under new rules."
Going through:
* Balancing traditional SEO and LLM SEO
* How LLMs read and process content
* What LLMs actually reward
* Tracking AI impact
And more! A must-read: https://t.co/m1eflCCfus
Number of Likes: 116
Number of Views: 6648
Number of Retweets: 22
Number of Replies: 3
Number of Quotes: 0
Number of Bookmarks: 116
Language: en
Tagged Users: Vercel, Malte Ubl
Hashtags: nan

Creation Date: 2025-06-12 15:47:22+00:00 
Tweet Text: 🚨 The next #SEOFOMO Meetup will be in a 🔥 new location in Europe and will be officially announced tomorrow 👀  Stay tuned for the announcement over here! 🙌 Registration will be free again... at a first-come, first-served basis, so you gotta be fast 💪 Stay tuned! https://t.co/qlo8TSD2Iz
Number of Likes: 11
Number of Views: 1248
Number of Retweets: 2
Number of Replies: 0
Number of Quotes: 0
Number of Bookmarks: 1
Language: en
Tagged Users: nan
Hashtags: SEOFOMO

Creation Date: 2025-06-12 09:00:59+00:00 
Tweet Text: AI search behavior research & analysis has now been made possible at scale thanks to @tryprofound  new Conversation Explorer 🤖🙌
What topics are searched in the most popular and relevant prompts about your business, products, or services? You can now do this research with Profound.
Read more: https://t.co/aSzxdF7Zto
PS: I love how tools like @tryprofound  and @Similarweb  are releasing data and functionality that allows us to better understand prompt behavior to assess how users are searching, how we're performing and easily identify opportunities. I'll be recording a video about it this week 💪
Number of Likes: 62
Number of Views: 5670
Number of Retweets: 22
Number of Replies: 6
Number of Quotes: 0
Number of Bookmarks: 42
Language: en
Tagged Users: Similarweb, Profound
Hashtags: nan


Example Non-Finfluencer Profile 2
- Profile Image: https://pbs.twimg.com/profile_images/1891047932851335168/PIAIeFfU_normal.jpg
- Profile Name: Money Quotes
- Profile ID: MoneyQuotesX
- Location: nan
- Profile Description: I’m passionate about sharing insightful, motivational quotes on money, wealth, and financial wisdom.
- Profile External Link: https://t.co/bkFjEmuxHw
- Profile Creation Date: 2024-01-30T02:32:36.000000Z
- Verified Profile: False
- Blue Verified Profile: True
- Protected Profile: False
- Number of Followers: 23561 Followers
- Following: 8923 Users
- Total Number of Tweets: 83017
- Number of Favorites: 162718
- Number of Media Content: 965
- Tweets (Sorted from Newest to Oldest):
Creation Date: 2025-06-20 10:46:47+00:00 
Tweet Text: There are two kinds of people:
- Those who think money is evil
- Those who understand how the world works
Number of Likes: 126
Number of Views: 6436
Number of Retweets: 18
Number of Replies: 40
Number of Quotes: 0
Number of Bookmarks: 9
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-19 16:05:20+00:00 
Tweet Text: In a world addicted to spending,
Discipline is a money-printing machine.
Number of Likes: 376
Number of Views: 18482
Number of Retweets: 42
Number of Replies: 100
Number of Quotes: 3
Number of Bookmarks: 33
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-19 11:07:01+00:00 
Tweet Text: There’s no glow-up like getting your finances right.
Number of Likes: 520
Number of Views: 40414
Number of Retweets: 70
Number of Replies: 98
Number of Quotes: 3
Number of Bookmarks: 29
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-18 16:53:43+00:00 
Tweet Text: Every rich person was once just someone who believed they could escape the script.
Number of Likes: 1505
Number of Views: 49862
Number of Retweets: 220
Number of Replies: 145
Number of Quotes: 5
Number of Bookmarks: 220
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-18 11:24:30+00:00 
Tweet Text: The real flex is making money doing what you’d do for free.
Number of Likes: 770
Number of Views: 1698619
Number of Retweets: 99
Number of Replies: 149
Number of Quotes: 7
Number of Bookmarks: 48
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-17 16:58:06+00:00 
Tweet Text: Financial literacy is your superpower in a system designed to keep you powerless.
Number of Likes: 1156
Number of Views: 46654
Number of Retweets: 193
Number of Replies: 116
Number of Quotes: 7
Number of Bookmarks: 186
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-17 11:04:29+00:00 
Tweet Text: Broke people gossip. Wealthy people strategize.
Number of Likes: 1345
Number of Views: 159473
Number of Retweets: 240
Number of Replies: 169
Number of Quotes: 10
Number of Bookmarks: 158
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-16 16:02:58+00:00 
Tweet Text: You don’t need 6 streams of income.
You need 1 that works.
1 stream that compounds.
1 stream that scales.
Multiple streams come after mastery.
Number of Likes: 444
Number of Views: 22462
Number of Retweets: 52
Number of Replies: 109
Number of Quotes: 2
Number of Bookmarks: 64
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-16 10:09:37+00:00 
Tweet Text: The real flex: waking up when you want, doing what you love, and never checking the price tag.
Number of Likes: 1144
Number of Views: 3471232
Number of Retweets: 174
Number of Replies: 188
Number of Quotes: 11
Number of Bookmarks: 133
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-15 15:49:19+00:00 
Tweet Text: The most dangerous money myth:
“I’ll start saving when I make more.”
Truth:
You won’t.
You’ll spend more.
Discipline doesn’t come with income.
It comes with mindset.
Number of Likes: 325
Number of Views: 13620
Number of Retweets: 45
Number of Replies: 94
Number of Quotes: 3
Number of Bookmarks: 25
Language: en
Tagged Users: nan
Hashtags: nan


Example Non-Finfluencer Profile 3
- Profile Image: https://pbs.twimg.com/profile_images/1588693184623693824/NtTR8aBz_normal.jpg
- Profile Name: Aviva - Denver - Warehouse
- Profile ID: AvivaRealEstate
- Location: Denver, CO
- Profile Description: #1 Commercial Real Estate Page Online | Managing Broker at Warehouse Hotline | DENVER, CO | Host of Commercial Real Estate Secrets Podcast
- Profile External Link: https://t.co/APLYn5WKDH
- Profile Creation Date: 2021-08-01T23:03:04.000000Z
- Verified Profile: False
- Blue Verified Profile: True
- Protected Profile: False
- Number of Followers: 9840 Followers
- Following: 1916 Users
- Total Number of Tweets: 5446
- Number of Favorites: 18767
- Number of Media Content: 903
- Tweets (Sorted from Newest to Oldest):
Creation Date: 2025-06-20 03:33:05+00:00 
Tweet Text: This flower blooms one night per year https://t.co/ypksU2zIYr
Number of Likes: 5
Number of Views: 450
Number of Retweets: 0
Number of Replies: 2
Number of Quotes: 0
Number of Bookmarks: 0
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-20 00:15:44+00:00 
Tweet Text: Tell me something better than having someone’s location on your iPhone
Number of Likes: 10
Number of Views: 4273
Number of Retweets: 1
Number of Replies: 10
Number of Quotes: 1
Number of Bookmarks: 1
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-19 16:22:51+00:00 
Tweet Text: On a call with @MrDavidSchwartz 
‘X is LinkedIn 10 years ago’
Agree or disagree?
Number of Likes: 0
Number of Views: 806
Number of Retweets: 0
Number of Replies: 3
Number of Quotes: 0
Number of Bookmarks: 0
Language: en
Tagged Users: OffMarketKing
Hashtags: nan

Creation Date: 2025-06-19 11:13:00+00:00 
Tweet Text: If you think retail is dead, you’re not paying attention to the drive-thrus printing money. https://t.co/uGMXtYdv3F
Number of Likes: 3
Number of Views: 438
Number of Retweets: 0
Number of Replies: 1
Number of Quotes: 0
Number of Bookmarks: 1
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-18 16:15:22+00:00 
Tweet Text: 🎉 CRE Secrets turns 2!
From square-foot talks to million-dollar mindsets—thank YOU for listening, sharing, and shaping the journey.
More insights. More wins. Year 3 is just getting started. 
Full article: https://t.co/wXTBWSMRDQ https://t.co/YitW84u6fJ
Number of Likes: 7
Number of Views: 400
Number of Retweets: 0
Number of Replies: 1
Number of Quotes: 0
Number of Bookmarks: 0
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-17 12:02:07+00:00 
Tweet Text: $100M raised—without going viral or gaming the system. Just smart syndication, long-term trust, and a clear investment thesis. Watch the full episode to learn how Clint Harris built lasting credibility—and scaled with integrity. https://t.co/WfUGeEnpzS https://t.co/CqfyU5ckOp
Number of Likes: 7
Number of Views: 689
Number of Retweets: 0
Number of Replies: 0
Number of Quotes: 0
Number of Bookmarks: 0
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-16 11:33:51+00:00 
Tweet Text: Forget the flip-and-sell mindset. Clint Harris shares how long-term real estate holds unlock time, location, and true financial freedom. https://t.co/Pqc4SLFr0N
Number of Likes: 10
Number of Views: 1492
Number of Retweets: 0
Number of Replies: 3
Number of Quotes: 0
Number of Bookmarks: 4
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-14 19:03:44+00:00 
Tweet Text: The wealthier the buyer / the lower the threshold for verifying finances.
Example = mom and pop tenant in 1920 SF. Need to see financials.
50MM land sale. An article will do.
Number of Likes: 4
Number of Views: 817
Number of Retweets: 0
Number of Replies: 0
Number of Quotes: 0
Number of Bookmarks: 0
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-12 18:55:54+00:00 
Tweet Text: How ironic is it that new investors often choose multifamily as a starting place in Commercial Real Estate.
If only they knew it was the biggest pain in the ass product type you could imagine.
Aside from hospitality.
Number of Likes: 168
Number of Views: 23417
Number of Retweets: 3
Number of Replies: 40
Number of Quotes: 0
Number of Bookmarks: 16
Language: en
Tagged Users: nan
Hashtags: nan

Creation Date: 2025-06-12 14:18:03+00:00 
Tweet Text: Property values are down—but your tax bill isn’t. Here’s why (and what to do about it). https://t.co/9Fk2Ct2sU3
Number of Likes: 2
Number of Views: 606
Number of Retweets: 0
Number of Replies: 2
Number of Quotes: 0
Number of Bookmarks: 1
Language: en
Tagged Users: nan
Hashtags: nan
"""

expert_reflection_prompt_template = """Expert Reflections from Professional Portfolio Manager
{expert_reflection_portfoliomanager}

Expert Reflections from Investment Advisor
{expert_reflection_investmentadvisor}

Expert Reflections from Chartered Financial Analyst
{expert_reflection_financialanalyst}

Expert Reflections from Economist
{expert_reflection_economist}"""

tiktok_finfluencer_onboarding_system_prompt = (
    base_finfluencer_onboarding_system_prompt.format(
        platform="Tiktok",
        finfluencer_examples=tiktok_finfluencer_examples,
        nonfinfluencer_examples=tiktok_nonfinfluencer_examples,
        expert_reflection_prompt_template=expert_reflection_prompt_template,
        profile_prompt_template=tiktok_profile_prompt_template,
    )
)

x_finfluencer_onboarding_system_prompt = (
    base_finfluencer_onboarding_system_prompt.format(
        platform="X (formerly Twitter)",
        finfluencer_examples=x_finfluencer_examples,
        nonfinfluencer_examples=x_nonfinfluencer_examples,
        expert_reflection_prompt_template=expert_reflection_prompt_template,
        profile_prompt_template=x_profile_prompt_template,
    )
)

base_finfluencer_onboarding_user_prompt = """You will be presented with a series of questions related to the profile of the {platform} user. Each question is preceded by predefined response options, each labeled with a symbol (e.g. "A1", "A2", "B1", etc.).

For each question, follow these instructions strictly:
1) Select the most likely response based strictly on the provided profile data. The chosen response must be the most accurate representation of the profile.
2) Select only one symbol/category per question. A title, symbol, and category cannot appear more than once in your answer.
3) Present the selected symbol for each question (if applicable) and write out in full the response associated with the selected symbol.
4) For each selected symbol/category, indicate the level of speculation involved in this selection on a scale from 0 (not speculative at all, every single element of the profile data was useful in the selection) to 100 (fully speculative, there is no information related to this question in the profile data). Speculation levels should be a direct measure of the amount of useful information available in the profile and pertain only to the information available in the profile data -- namely the username, name, description, profile picture, and videos from the profile-- and should not be affected by additional information available to you from any other source.

To ensure consistency, use the following guidelines to determine speculation levels:
0-20 (Low speculation): The profile data provides clear and direct information relevant to the question (e.g., explicit mention in the profile or videos).
21-40 (Moderate-low speculation): The profile data provides indirect but strong indicators relevant to the question (e.g., context from multiple sources within the profile or videos).
41-60 (Moderate speculation): The profile data provides some hints or partial information relevant to the question (e.g., inferred from user interests or indirect references).
61-80 (Moderate-high speculation): The profile data provides limited and weak indicators relevant to the question (e.g., very subtle hints or minimal context).
81-100 (High speculation): The profile data provides no or almost no information relevant to the question (e.g., assumptions based on very general information).

5) For each selected category, please explain at length what features of the data contributed to your choice and your speculation level.
6) Preserve a strictly structured response format to ensure clarity and ease parsing of the text.

Format your output as follows (this is just an example; do not focus on the specific explanation, symbol, category, speculation, or value provided):
**question: Indicate on a scale of 0 to 100, how likely this creator is a finfluencer (0 means most definitely not a finfluencer and 100 means most definitely a finfluencer)?**  
**explanation: [Detailed explanation for selected response]**  
**value: [Value selected]**  
**speculation: [Speculation score selected]**

**question: Indicate on a scale of 0 to 100, how influential this influencer is (0 means not at all influential and 100 means very influential with millions of followers and mainstream recognition)?**  
**explanation: [Detailed explanation for selected response]**  
**value: [Value selected]**  
**speculation: [Speculation score selected]**

**question: Indicate on a scale of 0 to 100, how credible or authoritative this influencer is (0 means not at all credible or authoritative and 100 means very credible and authoritative)?**  
**explanation: [Detailed explanation for selected response]**  
**value: [Value selected]**  
**speculation: [Speculation score selected]**

**question: Which of these areas of finance are the primary focus of the influencer’s posts?**  
**explanation: [Detailed explanation for selected response]**  
**symbol: [Symbol selected]**  
**category: [Category selected]**  
**speculation: [Speculation score selected]**

**question: Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's individual stock predictions (0 means very low quality and 100 means very high quality)?**  
**explanation: [Detailed explanation for selected response]**  
**value: [Value selected]**  
**speculation: [Speculation score selected]**

**question: Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's evaluation of market sentiment (0 means very low quality and 100 means very high quality)?**  
**explanation: [Detailed explanation for selected response]**  
**value: [Value selected]**  
**speculation: [Speculation score selected]**

**question: Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's broader evaluation of the economy (0 means very low quality and 100 means very high quality)?**  
**explanation: [Detailed explanation for selected response]**  
**value: [Value selected]**  
**speculation: [Speculation score selected]**

**question: Who is the influencer’s target audience?**  
**explanation: [Detailed explanation for selected response]**  
**symbol: [Symbol selected]**  
**category: [Category selected]**  
**speculation: [Speculation score selected]**

YOU MUST GIVE AN ANSWER FOR EVERY QUESTION!

Question 1: Indicate on a scale of 0 to 100, how likely this creator is a finfluencer (0 means most definitely not a finfluencer and 100 means most definitely a finfluencer)?

Question 2: Indicate on a scale of 0 to 100, how influential this influencer is (0 means not at all influential and 100 means very influential with millions of followers and mainstream recognition)? Please consider quantitative thresholds such as follower count and engagement rate when answering this question. For example, a micro-influencer will be in the 20-40 range, whereas an account with hundreds of thousands of followers and high engagement might rate 80+.

Question 3: Indicate on a scale of 0 to 100, how credible or authoritative this influencer is (0 means not at all credible or authoritative and 100 means very credible and authoritative)? For example, an experienced analyst with a solid track record and formal education is considered more credible than an influencer making unverified claims and promoting speculative bets without disclaimers.

Question 4: Which of these areas of finance are the primary focus of the influencer’s posts? Pick the most dominant theme or list two if it is truly split; however, one primary focus is always more preferable.
A1) Stock Trading and Equities: Content centered on stock picks, technical analysis, trading strategies, and market indexes.
A2) Bonds and Fixed Income: Content centered on interest rates, bond markets (government bonds, corporate bonds), Fed’s interest rate changes, and yield.
A3) Options Trading and Derivatives: Content centered on option strategies (calls, puts, spreads) and futures or other derivatives.
A4) Macroeconomic Analysis: Content centered on big-picture financial commentary – covering economic indicators, central bank policies, inflation, GDP, economic reports, and how world events affect markets.
A5) Cryptocurrency: Content centered on crypto assets (Bitcoin, Ethereum, altcoins, blockchain projects, NFTs), price updates, blockchain technology, certain crypto tokens, crypto trading tips, and news (regulatory updates and major moves in crypto markets).
A6) Real Estate Investments: Content centered on property investing, rental income, house flipping, REITs, housing market trends, share housing market data, and tips on evaluating real estate deal.
A7) Personal Finances and Budgeting: Content centered on personal finance topics such as budgeting, saving, debt management, credit scores, and general financial literacy without specific market picks.
A8) News and Current Events: Content centered on general news, current events, and trending topics that may not be directly related to finance but are discussed in the context of financial markets.
A9) Other: Content that does not fit into any of the above categories, such as lifestyle, entertainment, or unrelated topics.

Question 5: Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's individual stock predictions (0 means very low quality and 100 means very high quality)?

Question 6: Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's evaluation of market sentiment (0 means very low quality and 100 means very high quality)?

Question 7: Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's broader evaluation of the economy (0 means very low quality and 100 means very high quality)?

Question 8: Who is the influencer’s target audience?
B1) Young Investors
B2) Retirement Investors
B3) Seasoned Investors
B4) Others
"""

tiktok_finfluencer_onboarding_user_prompt = (
    base_finfluencer_onboarding_user_prompt.format(platform="Tiktok")
)

x_finfluencer_onboarding_user_prompt = base_finfluencer_onboarding_user_prompt.format(
    platform="X (formerly Twitter)"
)


# Market Signals Interview Prompts
base_finfluencer_interview_system_prompt = """Please put yourself in the shoes of a {platform} financial influencer participating in a financial market survey. Your profile was previously evaluated by an LLM during an onboarding phase and determined to be a financial influencer focusing on stock trading and equities, bonds and fixed income, or options trading and derivatives, based on your past video content and profile information. As part of this survey:
1. Your profile and videos will be monitored daily
2. You will undergo daily interviews to discuss your perspective on the financial markets

The details of your {platform} profile are as follows:
{profile_prompt_template}

Instructions
Answer the following questions based strictly on the available data while maintaining the persona and perspective of the {platform} financial influencer profile provided. Do not infer or assume any details beyond what is given. Keep responses concise, precise and data-driven."""

tiktok_finfluencer_interview_system_prompt = (
    base_finfluencer_interview_system_prompt.format(
        platform="TikTok",
        profile_prompt_template=tiktok_profile_prompt_template,
    )
)

x_finfluencer_interview_system_prompt = base_finfluencer_interview_system_prompt.format(
    platform="X (formerly Twitter)",
    profile_prompt_template=x_profile_prompt_template,
)

finfluencer_interview_user_prompt = """You will be presented with a series of questions, each preceded by predefined response options labeled with a symbol (e.g. "A1", "A2", "B1", etc.).

For each question, follow these instructions strictly:
1) Select the most likely response based strictly on the provided profile data. The chosen response must be the most accurate representation of the profile.
2) Select only one symbol/category per question. A title, symbol, and category cannot appear more than once in your answer.
3) Present the selected symbol for each question (if applicable) and write out in full the response associated with the selected symbol.
4) For each selected symbol/category, indicate the level of speculation involved in this selection on a scale from 0 (not speculative at all, every single element of the profile data was useful in the selection) to 100 (fully speculative, there is no information related to this question in the profile data). Speculation levels should be a direct measure of the amount of useful information available in the profile and pertain only to the information available in the profile data -- namely the username, name, description, profile picture, and videos from the profile-- and should not be affected by additional information available to you from any other source.

To ensure consistency, use the following guidelines to determine speculation levels:
0-20 (Low speculation): The profile data provides clear and direct information relevant to the question. (e.g., explicit mention in the profile or videos)
21-40 (Moderate-low speculation): The profile data provides indirect but strong indicators relevant to the question. (e.g., context from multiple sources within the profile or videos)
41-60 (Moderate speculation): The profile data provides some hints or partial information relevant to the question. (e.g., inferred from user interests or indirect references)
61-80 (Moderate-high speculation): The profile data provides limited and weak indicators relevant to the question. (e.g., very subtle hints or minimal context)
81-100 (High speculation): The profile data provides no or almost no information relevant to the question. (e.g., assumptions based on very general information)

5) For each selected category, please explain at length what features of the data contributed to your choice and your speculation level.
6) Preserve a strictly structured response format to ensure clarity and ease parsing of the text.

Format your output as follows (this is just an example; do not focus on the specific explanation, symbol, category, speculation, or value provided):
**question: Do you agree or disagree with the following statement: The U.S. economy is likely to enter a recession in the next 12 months?**
**explanation: [Detailed explanation for selected response]**
**symbol: [Symbol selected]**
**category: [Category selected]**
**speculation: [Speculation score selected]**

**question: How would you describe the current market sentiment among investors? Is sentiment very bearish, bearish, neutral, bullish, or very bullish?**
**explanation: [Detailed explanation for selected response]**
**symbol: [Symbol selected]**
**category: [Category selected]**
**speculation: [Speculation score selected]**

**question: Regarding the future direction of the stock market, are you very bearish, bearish, neutral, bullish, or very bullish?**
**explanation: [Detailed explanation for selected response]**
**symbol: [Symbol selected]**
**category: [Category selected]**
**speculation: [Speculation score selected]**

**question: In the next 1–3 months, do you expect U.S. stock market indices to rise, stay about the same, or fall?**
**explanation: [Detailed explanation for selected response]**
**symbol: [Symbol selected]**
**category: [Category selected]e**
**speculation: [Speculation score selected]**

**question: In the next 1–3 months, do you expect U.S. bond prices (or interest rates) to rise, remain unchanged, or fall?**
**explanation: [Detailed explanation for selected response]**
**symbol: [Symbol selected]**
**category: [Category selected]**
**speculation: [Speculation score selected]**

**question: Considering current market conditions, which sectors do you believe are poised to do well in the next 3–6 months? Can you give some background on these choices? Can you briefly explain why you picked these? Are there others you really think are outperformers? Why?**
**response: [Detailed response]**
**speculation: [Speculation score selected]**

**question: Considering current market conditions, which sectors do you believe are poised to do poorly in the next 3–6 months? Can you give some background on these choices? Can you briefly explain why you picked these? Are there others you really think are underperformers? Why?**
**response: [Detailed response]**
**speculation: [Speculation score selected]**

**question: Did you mention any stocks or stock tickers in the Russell 4000 list?**
**explanation: [Detailed explanation for selected response]**
**symbol: [Symbol selected]**
**category: [Category selected]e**
**speculation: [Speculation score selected]**

**question: Is there anything else about the economy or markets that you’d like to comment on that we didn’t cover?**
**response: [Detailed response]**
**speculation: [Speculation score selected]**

YOU MUST GIVE AN ANSWER FOR EVERY QUESTION WHILE MAINTAINING THE PERSONA AND PERSPECTIVE OF THE FINANCIAL INFLUENCER PROFILE PROVIDED!

Question 1: Do you agree or disagree with the following statement: The U.S. economy is likely to enter a recession in the next 12 months?
A1) Strongly Disagree
A2) Disagree
A3) Neither Agree/Disagree
A4) Agree
A5) Strongly Agree

Question 2: How would you describe the current market sentiment among investors? Is sentiment very bearish, bearish, neutral, bullish, or very bullish?
B1) Very Bearish
B2) Bearish
B3) Neutral
B4) Bullish
B5) Very Bullish

Question 3: Regarding the future direction of the stock market, are you very bearish, bearish, neutral, bullish, or very bullish?
C1) Very Bearish
C2) Bearish
C3) Neutral
C4) Bullish
C5) Very Bullish

Question 4: In the next 1–3 months, do you expect U.S. stock market indices to rise, stay about the same, or fall?
D1) Rise
D2) Stay About The Same
D3) Fall

Question 5: In the next 1–3 months, do you expect U.S. bond prices (or interest rates) to rise, remain unchanged, or fall?
E1) Rise
E2) Stay About The Same
E3) Fall

Question 6: Considering current market conditions, which sectors do you believe are poised to do well in the next 3–6 months? Can you give some background on these choices? Can you briefly explain why you picked these? Are there others you really think are outperformers? Why?

Question 7: Considering current market conditions, which sectors do you believe are poised to do poorly in the next 3–6 months? Can you give some background on these choices? Can you briefly explain why you picked these? Are there others you really think are underperformers? Why?

Question 8: Did you mention any stocks or stock tickers in the Russell 4000 list (e.g., {russell_4000_tickers})?
F1) Yes
F2) No

Question 9: Is there anything else about the economy or markets that you’d like to comment on that we didn’t cover?
"""

stock_recommendation_interview_user_prompt = """You will be presented with one of your previous posts and information about a particular stock/stock ticker that may or may not be mentioned in your post.

Based on the information provided to you, answer the following questions:
- mentioned_by_finfluencer: Confirm that you discussed or referenced this stock/stock ticker in your post (including tagged users and hashtags) by indicating Yes; otherwise indicate No.
- recommendation: Indicate on a scale of 0 to 100, your overall recommendation for this stock/stock ticker (0 means a very strong sell recommendation and 100 means a very strong buy recommendation). For example, a strong sell recommendation would be in the 0-20 range, a moderate sell recommendation would be in the 20-40 range, a hold recommendation would be in the 40-60 range, a moderate buy recommendation would be in the 60-80 range, and a strong buy recommendation would be 80+. Answer with NA, if you did not mention this stock/stock ticker in your post  (including tagged users and hashtags).
- explanation: Provide a brief explanation for your recommendation and the data features that contributed to your response.
- confidence: Indicate on a scale of 0 to 100, a measure of confidence for your stock recommendation (0-20 means low confidence, 20-40 means moderate-to-low confidence, 40-60 means moderate confidence, 60-80 means moderate-to-high confidence, and 80+ means high confidence). Answer with NA, if you did not mention this stock/stock ticker in your post (including tagged users and hashtags).
- virality: Indicate on a scale of 0 to 100, a measure of virality for your stock recommendation (0-20 means minimal virality, 20-40 means low virality, 40-60 means moderate virality, 60-80 means high virality, and 80+ means massive virality). Answer with NA, if you did not mention this stock/stock ticker in your post (including tagged users and hashtags).

Follow these instructions strictly when providing your response:
1) Select the most likely response based strictly on your provided profile data. The chosen response must be the most accurate representation of your profile.
2) Format your output as follows (this is just an example; do not focus on the specific response, explanation, recommendation, or value provided):
**mentioned_by_finfluencer: [Yes/No]**
**recommendation: [0-100 or NA if you did not mention this stock/stock ticker in your post (including tagged users and hashtags)]**
**explanation: [Detailed explanation for your selected response]**
**confidence: [0-100 or NA if you did not mention this stock/stock ticker in your post (including tagged users and hashtags)]**
**virality: [0-100 or NA if you did not mention this stock/stock ticker in your post (including tagged users and hashtags)]**

YOU MUST GIVE AN ANSWER TO EACH FIELD WHILE MAINTAINING THE PERSONA AND PERSPECTIVE OF THE FINANCIAL INFLUENCER PROFILE PROVIDED!

The stock mentioned in your post is as follows:
Stock Name: {stock_name}
Stock Ticker: {stock_ticker}
Post Date: {mention_date}
Post: {post}"""


# Election Polling
entity_geographic_inclusion_system_prompt = """You are analyzing a social media profile on TikTok to answer a set of questions. The TikTok profile data includes:
{profile_prompt}

Instructions
Analyze the provided information and answer the following questions based strictly on the available data. Do not infer or assume any details beyond what is given. Keep responses concise, precise and data-driven."""


entity_geographic_inclusion_user_prompt = """You will be presented with a series of questions related to the profile of the TikTok user. Each question is preceded by predefined response options, each labeled with a symbol (e.g. "A1", "A2", "B1", etc.).

For each question, follow these instructions strictly:
1) Select the most likely response based strictly on the provided profile data. The chosen response must be the most accurate representation of the profile.
2) Select only one symbol/category per question. A title, symbol, and category cannot appear more than once in your answer.
3) Present the selected symbol for each question (if applicable) and write out in full the response associated with the selected symbol.
4) For each selected symbol/category, indicate the level of speculation involved in this selection on a scale from 0 (not speculative at all, every single element of the profile data was useful in the selection) to 100 (fully speculative, there is no information related to this question in the profile data). Speculation levels should be a direct measure of the amount of useful information available in the profile and pertain only to the information available in the profile data -- namely the username, name, description, profile picture, and videos from the profile-- and should not be affected by additional information available to you from any other source.

To ensure consistency, use the following guidelines to determine speculation levels:
0-20 (Low speculation): The profile data provides clear and direct information relevant to the question. (e.g., explicit mention in the profile or videos)
21-40 (Moderate-low speculation): The profile data provides indirect but strong indicators relevant to the question. (e.g., context from multiple sources within the profile or videos)
41-60 (Moderate speculation): The profile data provides some hints or partial information relevant to the question. (e.g., inferred from user interests or indirect references)
61-80 (Moderate-high speculation): The profile data provides limited and weak indicators relevant to the question. (e.g., very subtle hints or minimal context)
81-100 (High speculation): The profile data provides no or almost no information relevant to the question. (e.g., assumptions based on very general information)

5) For each selected category, please explain at length what features of the data contributed to your choice and your speculation level.
6) Preserve a strictly structured response format to ensure clarity and ease parsing of the text.
For categorical questions, format your output as follows (this is just an example; do not focus on the specific question, symbol, or category):
**question: How well does the user engage with their audience?**
**explanation: [Detailed reasoning for selection]**
**symbol: A1)**
**category: Very Well**
**speculation: 90**

For numerical questions (0-100 scale), format your output as follows (this is just an example; do not focus on the specific question, symbol, or category):
**question: Indicate on a scale of 0 to 100, how knowledgable is the user about finance – 0 means not at all knowledgeable and 100 means very knowledgeable?**
**explanation: [Detailed reasoning for selection]**
**value: 20**
**speculation: 90**

For open-ended questions, format your output as follows (this is just an example; do not focus on the specific question, symbol, or category):
**question: Are there any stocks you expect to **underperform** or decline in the next 3-6 months?**
**response: [Detailed response]**
**speculation: 90**

YOU MUST GIVE AN ANSWER FOR EVERY QUESTION!

Question 1: Is this an account of a real-life existing person, or of another kind of entity?
A1) Person
A2) Other

Question 2: Does the user of this TikTok account live in Canada?
B1) Yes 
B2) No

Question 3: If the response to Question 2 is “Yes,” specify the state (province) the user is living in. Otherwise, respond with “NA.”"""


polling_system_prompt = """{profile_prompt}"""


polling_user_prompt = """"""
