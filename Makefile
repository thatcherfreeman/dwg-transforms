lint:
	clang-format -i RCM\ IDTs/*.dctl
	clang-format -i RCM\ ODTs/*.dctl

all: lint