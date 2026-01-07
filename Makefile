.PHONY: all lint format test build-ui verify

lint:
	$(MAKE) -C windwatts-api lint
	cd windwatts-ui && yarn lint

format:
	$(MAKE) -C windwatts-api format
	cd windwatts-ui && yarn format

test:
	$(MAKE) -C windwatts-api test
	cd windwatts-ui && yarn test -- --run

build-ui:
	cd windwatts-ui && yarn build

verify: lint test build-ui

