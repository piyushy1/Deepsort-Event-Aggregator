import re

ex_queries = [
"select * from within NUMFRAMEWINDOW(10)",
"select LEFT(object1, object2) from within NUMFRAMEWINDOW(10)"
]


## basic query parser to parse operator and frame window number

def find_operator(query):
	g = re.search(r'select (.*) from(.*)within(.*)', query, re.IGNORECASE)
	operator = g.group(1)
	if operator.strip() != '*':
		found = re.search(r'(.*)\(.*\)', operator, re.IGNORECASE).group(1)
		print(found)


def find_window(query):
	g = re.search(r'select (.*) from(.*)within(.*)', query, re.IGNORECASE)
	window = g.group(3)
	window_count = re.search(r'\(([0-9]*)\)', window).group(1)
	print(window_count)


for i in ex_queries:
	find_operator(i)
	find_window(i)