# tau Makefile
#
# Library-only Go module. The canonical `tau` binary is built from the
# separate tau-cli module (../tau-cli). This Makefile only carries the
# targets library consumers and SDK contributors need: test, lint, proto,
# tidy, fmt, vet, check, clean, help.

TEST_TIMEOUT := 60s

.PHONY: test e2e lint proto tidy clean fmt vet check help

# test: run unit + integration tests (excludes e2e by default)
test:
	go test -timeout $(TEST_TIMEOUT) ./...

# e2e: run end-to-end tests (agent loop only; CLI e2e lives in tau-cli)
e2e:
	TAU_RUN_E2E=1 go test -timeout 120s ./test/e2e/...

# lint: run golangci-lint with the project config
lint:
	golangci-lint run ./...

# proto: regenerate *.pb.go from internal/proto/plugin.proto
proto:
	protoc \
		--go_out=. --go_opt=module=github.com/taucentral/tau \
		--go-grpc_out=. --go-grpc_opt=module=github.com/taucentral/tau \
		internal/proto/plugin.proto

# tidy: sync go.mod/go.sum
tidy:
	go mod tidy

# fmt + vet: standalone formatters
fmt:
	gofmt -s -w .
	goimports -w -local github.com/taucentral/tau .

vet:
	go vet ./...

# check: fast pre-commit gate (fmt + vet + lint)
check: vet lint

# clean: remove build artifacts
clean:
	rm -rf bin coverage.out

help:
	@echo "tau Makefile targets (library-only; build/install/run live in tau-cli):"
	@echo "  test     - run unit + integration tests"
	@echo "  e2e      - run agent-loop end-to-end tests"
	@echo "  lint     - run golangci-lint"
	@echo "  proto    - regenerate *.pb.go from plugin.proto"
	@echo "  tidy     - go mod tidy"
	@echo "  fmt      - gofmt + goimports"
	@echo "  vet      - go vet"
	@echo "  check    - vet + lint"
	@echo "  clean    - remove build artifacts"
