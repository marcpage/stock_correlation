#!/usr/bin/env python3

""" Find and visualize correlation between various equities.
    Takes ticker symbols as parameters
"""

import os
import sys
import re
import time
import math
import urllib.request
import platform

import svgwrite  # pip install svgwrite


QUOTE_API = "https://query1.finance.yahoo.com/v7/finance/download/"
# 2000000000 means this will work until May, 17, 2033
QUOTE_URL = (
    QUOTE_API
    + "%(symbol)s?period1=0&period2=2000000000&interval=1d"
    + "&events=history&includeAdjustedClose=true"
)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) "
    + "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
)
STATS_URL = "https://finance.yahoo.com/quote/%(symbol)s"
YIELD_PATTERN = re.compile(r""""(dividendYield|yield)":{"raw":([0-9.]+),""")
EXPENSE_PATTERN = re.compile(r""""annualReportExpenseRatio":{"raw":([0-9.]+),""")
NET_ASSETS = re.compile(r"""(totalAssets|marketCap)":{"raw":([0-9.]+),""")
MAX_CIRCLE_RADIANS = 2.0 * 3.14159265


def cache_file_path(*parts):
    """creates a path to a temporary file that can be created."""
    (tm_year, tm_mon, tm_day, _, _, _, _, _, _) = time.localtime()
    parts = list(parts)
    parts.extend((tm_year, tm_mon, tm_day))

    if platform.system() == "Darwin":
        cache_dir = os.path.join(
            os.environ["HOME"], "Library", "Caches", os.path.split(__file__)[1]
        )
    elif platform.system() == "Linux":
        cache_dir = os.path.join(os.environ["HOME"], "." + os.path.split(__file__)[1])
    else:
        cache_dir = os.path.join(os.environ["TMP"], os.path.split(__file__)[1])

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    return os.path.join(cache_dir, "_".join([str(x) for x in parts]))


def get_url_contents(url, *name_parts):
    """Get the contents of a web page"""
    cache_path = cache_file_path(*name_parts)

    try:
        with open(cache_path, "rb") as cache_file:
            contents = cache_file.read()

    except FileNotFoundError:
        request = urllib.request.Request(
            url, data=None, headers={"User-Agent": USER_AGENT}
        )

        with urllib.request.urlopen(request) as connection:
            contents = connection.read()

        with open(cache_path, "wb") as cache_file:
            cache_file.write(contents)

    return contents


def get_symbol_history(symbol):
    """Get the history of an equity symbol"""
    return get_url_contents(QUOTE_URL % {"symbol": symbol}, "history", symbol).decode(
        "utf-8"
    )


def get_symbol_stats(symbol):
    """Get expense ratio and yield of an equity"""
    contents = get_url_contents(STATS_URL % {"symbol": symbol}, "stats", symbol).decode(
        "utf-8"
    )
    has_expense_ratio = EXPENSE_PATTERN.search(contents)
    return {
        "yield": float(YIELD_PATTERN.search(contents).group(2)),
        "expense_ratio": float(has_expense_ratio.group(1) if has_expense_ratio else 0.0),
        "total_value": float(NET_ASSETS.search(contents).group(2)),
    }


def load_history(symbol):
    """Get this history as a dictionary of date to information on that date"""
    contents = get_symbol_history(symbol)
    lines = contents.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")
    fields = lines.pop(0).split(",")
    dates = [dict(zip(fields, x.split(","))) for x in lines]
    return {x["Date"]: x for x in dates}


def date_to_seconds(date_str):
    """Convert date to seconds"""
    return time.mktime(time.strptime(date_str, "%Y-%m-%d"))


def calculate_variance(history, stats):
    """Compare the histories of all symbols and get their variance from line fit"""
    mean_date = sum([date_to_seconds(d) for d in history]) / len(history)
    mean_adj_close = sum([float(history[d]["Adj Close"]) for d in history]) / len(
        history
    )
    product_sum = sum(
        [
            (date_to_seconds(d) - mean_date) * (float(history[d]["Adj Close"]))
            for d in history
        ]
    )
    date_square_sum = sum([(date_to_seconds(d) - mean_date) ** 2 for d in history])
    slope = product_sum / date_square_sum
    y_intercept = mean_adj_close - slope * mean_date

    for date in history:
        expected_adj_close = slope * date_to_seconds(date) + y_intercept
        actual_value = float(history[date]["Adj Close"])
        history[date]["variance"] = (
            actual_value - expected_adj_close
        ) / expected_adj_close

    # normalize variances (0% to 100%)
    min_variance = min([history[d]["variance"] for d in history])
    max_variance = max([history[d]["variance"] for d in history])

    for date in history:
        history[date]["std_variance"] = (history[date]["variance"] - min_variance) / (
            max_variance - min_variance
        )

    result = {'history': history, 'slope': slope * 60 * 60 * 24 * 365 / mean_adj_close}
    result.update({'stats': stats})
    return result


def calculate_distance(history1, history2, key="variance"):
    """Determine how much two histories varies"""
    overalapping_dates = [d for d in history1 if d in history2]
    square_sum = 0.0

    for date in overalapping_dates:
        square_sum += (history1[date][key] - history2[date][key]) ** 2

    return math.sqrt(square_sum)


class Point:
    """A point in 2D space"""

    def __init__(self, x, y):
        """create a new point"""
        (self.__x, self.__y) = (
            x,
            y,
        )

    def __add__(self, vector):
        """Add a vector onto a point"""
        return Point(vector.get_dx() + self.__x, vector.get_dy() + self.__y)

    def __sub__(self, point):
        """Find a vector between two points"""
        return Vector(self.get_x() - point.get_x(), self.get_y() - point.get_y())

    def __str__(self):
        """Display the point"""
        return "(%0.2f, %0.2f)" % (self.__x, self.__y)

    def __repr__(self):
        """display the point"""
        return str(self)

    def get_x(self):
        """Get X coordinate"""
        return self.__x

    def get_y(self):
        """Get Y coordinate"""
        return self.__y


class Vector:
    """A vector in 2D space"""

    def __init__(self, dx, dy):
        """create a vector"""
        (self.__dx, self.__dy) = (dx, dy)

    def __add__(self, vector):
        """Add two vectors"""
        return Vector(self.get_dx() + vector.get_dx(), self.get_dy() + vector.get_dy())

    def __str__(self):
        """display the vector"""
        return "[%0.2f, %0.2f]" % (self.__dx, self.__dy)

    def __repr__(self):
        """display the vector"""
        return str(self)

    def get_dx(self):
        """Get the change in X direction"""
        return self.__dx

    def get_dy(self):
        """Get the change in Y direction"""
        return self.__dy

    def magnitude(self):
        """Get the magnitude of the vector"""
        return math.sqrt(self.__dx ** 2 + self.__dy ** 2)

    def scaled(self, factor):
        """Scale the vector"""
        return Vector(factor * self.__dx, factor * self.__dy)


def add_distances(histories):
    """Calculate the distance (difference in variance) between all equities"""
    for symbol in histories:
        histories[symbol]["distance"] = {
            s: calculate_distance(
                histories[symbol]["history"], histories[s]["history"], "variance"
            )
            for s in histories
            if s != symbol
        }
        histories[symbol]["std_distance"] = {
            s: calculate_distance(
                histories[symbol]["history"], histories[s]["history"], "std_variance"
            )
            for s in histories
            if s != symbol
        }


def movement(symbol1, symbol2, points, histories):
    """Move symbol1 towards the expected distance from symbol2"""
    distance = points[symbol2] - points[symbol1]
    distance_magnitude = distance.magnitude()
    expected_distance = histories[symbol1]["std_distance"][symbol2]
    return (
        distance.scaled((distance_magnitude - expected_distance) / distance_magnitude)
        if distance_magnitude > 0
        else Vector(0.0, 0.0)
    )


def apply_gravity(points, histories, speed=0.10):
    """Move all points towards their expected distances from all other points"""
    velocities = {s: Vector(0, 0) for s in histories}
    largest_velocity = Vector(0, 0)

    for symbol1 in histories:
        for symbol2 in [s for s in histories if s != symbol1]:
            distance_to_expected = movement(symbol1, symbol2, points, histories)
            velocities[symbol1] += distance_to_expected.scaled(speed / 2.0)

    for symbol in points:
        points[symbol] = points[symbol] + velocities[symbol]

        if velocities[symbol].magnitude() > largest_velocity.magnitude():
            largest_velocity = velocities[symbol]

    return largest_velocity.magnitude()


def bubble_color(expense_ratio, min_expense_ratio, max_expense_ratio, slope, min_slope, max_slope):
    min_saturation = 0.80
    red = int(
            255
            * (expense_ratio - min_expense_ratio)
            / (max_expense_ratio - min_expense_ratio)
        ) if max_expense_ratio > min_expense_ratio else 128
    green = int(
            255
            * (max_expense_ratio - expense_ratio)
            / (max_expense_ratio - min_expense_ratio)
        ) if max_expense_ratio > min_expense_ratio else 128
    blue = 0
    saturation = (
        (slope - min_slope)
        / (max_slope - min_slope)) if max_slope > min_slope else 0.50
    return "#%02x%02x%02x" % (
        red + int((255 - red) * min_saturation * (1.00 - saturation)),
        green + int((255 - green) * min_saturation * (1.00 - saturation)),
        blue + int((255 - blue) * min_saturation * (1.00 - saturation))
    )


def add_circle(drawing, main_drawing, location, radius, color):
    drawing.add(
        main_drawing.circle(
            center=location,
            r=radius,
            fill=color,
        )
    )


def add_label(drawing, main_drawing, location, text, rotate=0, size="1px"):
    drawing.add(
        main_drawing.text(
            text,
            insert=location,
            font_size=size,
            transform='rotate(%d,%s, %s)' % (rotate, location[0], location[1]),
        )
    )


def add_rect(drawing, main_drawing, x, y, width, height, color):
    drawing.add(
        main_drawing.rect((x, y), (width, height), fill=color)
    )


def graph_key(drawing, main_drawing, width, height, radius_info, color_info, saturation_info):
    max_radius = radius_info['max']
    min_radius = radius_info['min']
    mid_radius = (max_radius + min_radius) / 2
    max_yield = 100.0 * radius_info['max_value']
    min_yield = 100.0 * radius_info['min_value']
    mid_yield = (max_yield + min_yield) / 2
    max_expense_ratio = color_info['max_value']
    min_expense_ratio = color_info['min_value']
    mid_expense_ratio = (max_expense_ratio + min_expense_ratio) / 2
    max_slope= saturation_info['max_value']
    min_slope = saturation_info['min_value']
    mid_slope = (max_slope + min_slope) / 2
    border = 0.5
    cell_size = 3.0
    color_table = [
        [
            bubble_color(max_expense_ratio, min_expense_ratio, max_expense_ratio, min_slope, min_slope, max_slope),
            bubble_color(mid_expense_ratio, min_expense_ratio, max_expense_ratio, min_slope, min_slope, max_slope),
            bubble_color(min_expense_ratio, min_expense_ratio, max_expense_ratio, min_slope, min_slope, max_slope),
        ],
        [
            bubble_color(max_expense_ratio, min_expense_ratio, max_expense_ratio, mid_slope, min_slope, max_slope),
            bubble_color(mid_expense_ratio, min_expense_ratio, max_expense_ratio, mid_slope, min_slope, max_slope),
            bubble_color(min_expense_ratio, min_expense_ratio, max_expense_ratio, mid_slope, min_slope, max_slope),
        ],
        [
            bubble_color(max_expense_ratio, min_expense_ratio, max_expense_ratio, max_slope, min_slope, max_slope),
            bubble_color(mid_expense_ratio, min_expense_ratio, max_expense_ratio, max_slope, min_slope, max_slope),
            bubble_color(min_expense_ratio, min_expense_ratio, max_expense_ratio, max_slope, min_slope, max_slope),
        ],
    ]

    circle_center = (width - max_radius - border, height - max_radius - border)
    add_circle(drawing, main_drawing, circle_center, max_radius, "#88AAFF")
    add_label(drawing, main_drawing, circle_center, "%0.2f%%"%(max_yield))

    circle_center = (width - 2 * max_radius - mid_radius - border, height - mid_radius - border)
    add_circle(drawing, main_drawing, circle_center, mid_radius, "#88AAFF")
    add_label(drawing, main_drawing, circle_center, "%0.2f%%"%(mid_yield))

    circle_center = (width - 2 * max_radius - 2 * mid_radius - min_radius - border, height - min_radius - border)
    add_circle(drawing, main_drawing, circle_center, min_radius, "#88AAFF")
    add_label(drawing, main_drawing, circle_center, "%0.2f%%"%(min_yield))

    circle_center = (width - 2 * max_radius - border, height - border)
    add_label(drawing, main_drawing, circle_center, "Yield")

    for row in range(0, len(color_table)):
        for column in range(0, len(color_table[row])):
            add_rect(drawing, main_drawing, border + column * cell_size, height - border - (row + 1) * cell_size, cell_size, cell_size, color_table[row][column])

    circle_center = (border + cell_size * 0 + border, height - border - cell_size * 3 - border)
    add_label(drawing, main_drawing, circle_center, "%0.2f%%"%(100.0 * max_expense_ratio), rotate=-30)

    circle_center = (border + cell_size * 1 + border, height - border - cell_size * 3 - border)
    add_label(drawing, main_drawing, circle_center, "%0.2f%%"%(100.0 * mid_expense_ratio), rotate=-30)

    circle_center = (border + cell_size * 2 + border, height - border - cell_size * 3 - border)
    add_label(drawing, main_drawing, circle_center, "%0.2f%%"%(100.0 * min_expense_ratio), rotate=-30)

    circle_center = (border + cell_size * 0, height - border - cell_size * 4 - border)
    add_label(drawing, main_drawing, circle_center, "Expense Ratio")

    circle_center = (border + cell_size * 3 + border, height - border - cell_size * 0 - cell_size / 2)
    add_label(drawing, main_drawing, circle_center, "%0.2f%%"%(100.0 * min_slope))

    circle_center = (border + cell_size * 3 + border, height - border - cell_size * 1 - cell_size / 2)
    add_label(drawing, main_drawing, circle_center, "%0.2f%%"%(100.0 * mid_slope))

    circle_center = (border + cell_size * 3 + border, height - border - cell_size * 2 - cell_size / 2)
    add_label(drawing, main_drawing, circle_center, "%0.2f%%"%(100.0 * max_slope))

    circle_center = (border + cell_size * 4 + 2 * border, height - border - cell_size * 2 - cell_size / 2)
    add_label(drawing, main_drawing, circle_center, "Growth Rate", rotate=90)


def graph_points(histories, points=None, scale=1):
    """Graph all the equities"""
    # pylint: disable=too-many-locals
    if points is None:
        points = {s: Point(*histories[s]["std_location"]) for s in histories}

    max_radius = min(
        [
            min(
                [
                    histories[s1]["std_distance"][s2]
                    for s2 in histories[s1]["std_distance"]
                ]
            )
            for s1 in histories
        ]
    )
    # sqrt because yield is radius
    min_yield = math.sqrt(min([histories[s]["stats"]["yield"] for s in histories]))
    max_yield = math.sqrt(max([histories[s]["stats"]["yield"] for s in histories]))
    min_expense_ratio = min([histories[s]["stats"]["expense_ratio"] for s in histories])
    max_expense_ratio = max([histories[s]["stats"]["expense_ratio"] for s in histories])
    min_slope = min([histories[s]["slope"] for s in histories])
    max_slope = max([histories[s]["slope"] for s in histories])
    min_radius = 0.25 * max_radius
    min_x = min([points[p].get_x() for p in points]) - 2 * max_radius
    max_x = max([points[p].get_x() for p in points]) + 2 * max_radius
    min_y = min([points[p].get_y() for p in points]) - 2 * max_radius
    max_y = max([points[p].get_y() for p in points]) + 2 * max_radius
    main_drawing = svgwrite.Drawing(
        size=(scale * (max_x - min_x), scale * (max_y - min_y))
    )
    drawing = main_drawing.g(transform="scale(%d)" % (scale))
    add_rect(drawing, main_drawing, 0, 0, max_x - min_x, max_y - min_y, "lightgray")
    graph_key(drawing, main_drawing, (max_x - min_x), (max_y - min_y),
        {'min': min_radius, 'max': max_radius, 'min_value': min_yield**2, 'max_value': max_yield**2},
        {'min_value': min_expense_ratio, 'max_value': max_expense_ratio},
        {'min_value': min_slope, 'max_value': max_slope})
    for symbol in points:
        expense_ratio = histories[symbol]["stats"]["expense_ratio"]
        slope = histories[symbol]["slope"]
        color = bubble_color(expense_ratio, min_expense_ratio, max_expense_ratio, slope, min_slope, max_slope)
        dividend = math.sqrt(histories[symbol]["stats"]["yield"])
        radius = (max_radius - min_radius) * (dividend - min_yield) / (
            max_yield - min_yield
        ) + min_radius
        add_circle(drawing, main_drawing, (points[symbol].get_x() - min_x, points[symbol].get_y() - min_y), radius, color)

    for symbol in points:
        add_label(drawing, main_drawing, (points[symbol].get_x() - min_x, points[symbol].get_y() - min_y), symbol)

    main_drawing.add(drawing)
    return main_drawing.tostring()


def add_locations(histories):
    """Place the equities in the edge of a circle, close to their nearest equity"""
    # pylint: disable=too-many-locals
    max_distance = max(
        [
            max(
                [
                    histories[s1]["std_distance"][s2]
                    for s2 in histories[s1]["std_distance"]
                ]
            )
            for s1 in histories
        ]
    )
    min_distance = min(
        [
            min(
                [
                    histories[s1]["std_distance"][s2]
                    for s2 in histories[s1]["std_distance"]
                ]
            )
            for s1 in histories
        ]
    )
    circle_radius = max_distance * (len(histories) - 1) / 2.0
    radians_per_point = MAX_CIRCLE_RADIANS / len(histories)
    symbols = list(histories)
    negative = True
    index = 0
    start_symbol = [
        s1
        for s1 in histories
        if min_distance
        == min(
            [histories[s1]["std_distance"][s2] for s2 in histories[s1]["std_distance"]]
        )
    ][0]
    points = {
        start_symbol: Point(
            math.cos(index * radians_per_point) * circle_radius,
            math.sin(index * radians_per_point) * circle_radius,
        )
    }
    symbols.remove(start_symbol)
    used_symbols = [start_symbol]

    while symbols:
        sign = -1 if negative else 1

        if negative:
            index += 1
            near_symbol = used_symbols[0]
            insert_location = 0
        else:
            near_symbol = used_symbols[-1]
            insert_location = len(used_symbols)

        next_symbol = sorted(
            symbols,
            key=lambda s: histories[near_symbol]["std_distance"][s],
        )[0]
        points[next_symbol] = Point(
            math.cos(sign * index * radians_per_point) * circle_radius,
            math.sin(sign * index * radians_per_point) * circle_radius,
        )

        negative = not negative
        symbols.remove(next_symbol)
        used_symbols.insert(insert_location, next_symbol)

    change = 100

    with open("log.html", "w") as log_file:
        log_file.write("<html><body>\n")

        while change > 0.001:
            change = apply_gravity(points, histories, speed=0.050)
            log_file.write(graph_points(histories, points) + "\n")
            log_file.flush()

        log_file.write("</body></html>\n")

    min_x = min([points[p].get_x() for p in points])
    min_y = min([points[p].get_y() for p in points])

    for symbol in points:
        histories[symbol]["std_location"] = (
            points[symbol].get_x() - min_x,
            points[symbol].get_y() - min_y,
        )


def main():
    """Plot various equities"""

    if len(sys.argv) <= 2:
        print("Usage: \n%s %s [symbol1] [symbol2] ..."%(sys.executable, __file__))
        print("You must specify at least two symbols")
        sys.exit(1)

    histories = {
        x: calculate_variance(load_history(x), get_symbol_stats(x))
        for x in sys.argv[1:]
    }

    add_distances(histories)
    add_locations(histories)

    with open("plot.html", "w") as plot_file:
        plot_file.write("<html>\n")
        plot_file.write("<head><style>td { text-align: right; }</style>\n")
        plot_file.write("<body>\n")
        plot_file.write(graph_points(histories, scale=20) + "\n")
        plot_file.write("<table>\n")
        plot_file.write("<tr><th>Symbol</th><th>Yield</th><th>Expense Ratio</th><th>Total Assets / Market Cap</th><th>Percent of total</th><th>Growth Rate</th><tr>\n")
        market_sum = sum([histories[x]['stats']['total_value'] for x in histories])
        for symbol in sorted(histories, key=lambda x:histories[x]['stats']['total_value'], reverse=True):
            plot_file.write("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n"%(
                symbol,
                "%0.2f%%"%(100.0 * histories[symbol]['stats']['yield']),
                "%0.2f%%"%(100.0 * histories[symbol]['stats']['expense_ratio']),
                "%0.0f"%(histories[symbol]['stats']['total_value']),
                "%0.2f%%"%(100.0 * histories[symbol]['stats']['total_value'] / market_sum),
                "%0.2f%%"%(100.0 * histories[symbol]['slope']),
            ))
        plot_file.write("</table>\n")
        plot_file.write("</body></html>\n")


if __name__ == "__main__":
    main()
