.PHONY: test

test:
	@go test -tags testsuite -cover -coverprofile=coverage.out -race  ./...
	@go tool cover -func=coverage.out
