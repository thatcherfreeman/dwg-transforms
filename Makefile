lint:
	clang-format -i RCM\ IDTs/*.dctl
	python lint_dctl.py "RCM IDTs/"
	clang-format -i RCM\ ODTs/*.dctl
	python lint_dctl.py "RCM ODTs/"
	clang-format -i Stills\ IDTs/*.dctl
	python lint_dctl.py "Stills IDTs/"

all: lint