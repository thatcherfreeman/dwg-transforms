lint:
	clang-format -i RCM\ IDTs/*.dctl
	clang-format -i RCM\ ODTs/*.dctl
	clang-format -i Stills\ IDTs/*.dctl

all: lint