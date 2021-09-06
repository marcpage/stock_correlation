# Stock Correlation

## Disclaimer

There are no warranties, express or implied, for this tool. This tool uses past market performance, which may have absolutely nothing to do with future performance. This tool is for educational and entertainment purposes only. Seek qualified investment professionals before you invest. Make sure you have learned everything you can from multiple perspectives before investing. Realize that any money you invest, you may lose all of it. Never trust some random guy on the internet to give you investing advice. Never trust "meme stonks" or the popular investment advice "out there." Any losses you incur are from your choices and not the use of this tool or following any ideas or theories from the author or on the internet. All investors are advised to conduct their own independent research into equities before making a purchase decision. In addition, investors are advised that past stock performance is no guarantee of future price appreciation. If you are still reading this paragraph, good on you, but seriously, seek competent advisors before investing.

## Summary

> "What [Markowitz Portfolio Theory] says is that the volatility/risk of a portfolio may differ dramatically from the volatility/risk of the portfolio’s components. In other words, you can have two assets with both high standard deviations and high potential returns, but when combined they give you a portfolio with modest standard deviation but the same high potential return.” 
> - Russell Wild

This tool is intended to help find a portfolio of equities (I prefer Market Capitalization Weighted Index ETFs, but this should work for anything) that are not correlated in their movements. In other words, when one is up, the other is down. The idea being that if you dollar cost average (buying the same dollar amount every month) into this portfolio, you will always being going in heavier into the "down" equity (buying low), and when you rebalance (selling high and buying low) there will always be an equity that is high and another that is low.

(If you are intrigued by this tool but most of the terms seem foreign to you, I am a financial coach and would be happy to discuss them. You can [book a free one-hour consultation](https://linktr.ee/resolvetoexcel).)

## Requirements

> pip3 install svgwrite

You will need to install *svgwrite* for this tool to work.

## Usage

> python3 stock_correlation.py VTI BND SCHD MGC

This will generate two files in the current directory, *log.html* and *plot.html*. The two *log.html* files show the progression. The *plot.html* file shows the final results.

## What it does

**tldr;** It plots stocks on a chart showing how closely their movements correlate.

The script will go out and gather the full *history* for each ticker symbol passed on the command line and also fetch its *yield* and *expense ratio* (from Yahoo Finance). It then does a basic line-fit for the history (using the Adjusted Close) and then determines what percent above or below the line-fit each day's close (again, Adjusted Close) is. We then normalize these swings into a percent. 

Once we have the swings normalized, we then determine the root-mean-square of the differences (over the same days) between every pair of equities. This gives us a correlation number (the higher the number, the less correlated the movements of the equities are, the lower the number the more correlated).

We then place each of the equities on the circumference of a circle, starting with the two most closely correlated equities and continue around the circle placing equities that are closest to the last placed equity. By the end we have a circle with closely related equities on the right and less closely related equities on the left.

We then go through several iterations of slowly moving each equity towards a position that will have it closer to more correlated equities and further from less correlated equities. Once the movement settles, we then create *plot.html* to show the final position. (*log.html* shows each step in the process from circle to final location).

<img width="889" alt="Screen Shot 2021-09-06 at 5 45 59 PM" src="https://user-images.githubusercontent.com/695749/132264286-ce37fcd2-df0a-477e-b272-864c04765fbf.png">

## How to read the chart

An example chart is shown below. The *ticker symbol* is displayed (lower left of the text is at the center of the circle). 

The *area of each circle* is relative to its *yield* (the smaller the circle, the lower the yield). 

The *color* of the circle shows the relative *expense ratio*. Bright red (🔴) is the highest expense ratio, bright green (🟢) is the lowest expense ratio. Darker colors are closer to being halfway between the highest and lowest.

The *distance* between circles represents how closely their movements correlate with eachother. The closer the circles are, the more closely their movements correlate with eachother.

<img width="944" alt="Screen Shot 2021-09-06 at 2 55 58 PM" src="https://user-images.githubusercontent.com/695749/132262283-421e8568-1b61-47f1-800c-6693c81316a2.png">

## How I use this tool

I grab a bunch of Market Capitalization Weighted Index ETFs and pass them all on the command line. I then look for reasons to remove some (simplify my portfolio). If there is a grouping, I remove the redder (higher expense ratio), smaller (lower dividend) ones. It is not an exact science, just gut feel. I then run the tool again with the new list. I keep going until it gets hard to justify removing anymore.
