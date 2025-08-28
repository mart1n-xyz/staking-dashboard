# SNT Staking Dashboard - Smart Contracts

This directory contains the Forge/Foundry smart contract development environment for the SNT Staking Dashboard custom multicall contract.

## Setup

The contracts environment is already configured and ready to use.

### Dependencies Installed:
- **Forge Std**: Testing and scripting utilities
- **OpenZeppelin Contracts v5.4.0**: Standard contract implementations

## Structure

```
contracts/
├── src/                    # Contract source files
├── test/                   # Test files
├── script/                 # Deployment scripts
├── lib/                    # Dependencies (forge-std, openzeppelin-contracts)
├── out/                    # Compiled contracts (gitignored)
└── foundry.toml           # Forge configuration
```

## Commands

```bash
# Build contracts
forge build

# Run tests
forge test

# Deploy to Sepolia
forge script script/Deploy.s.sol --rpc-url $SEPOLIA_RPC_URL --private-key $PRIVATE_KEY --broadcast --verify

# Check gas costs
forge test --gas-report
```

## Configuration

- **Solidity Version**: 0.8.23
- **Optimizer**: Enabled (200 runs)
- **Networks**: Sepolia, Mainnet
- **Verification**: Etherscan integration

## Environment Variables

Copy `../env.example` to `../.env` and configure:
- `SEPOLIA_RPC_URL`
- `MAINNET_RPC_URL` 
- `PRIVATE_KEY`
- `ETHERSCAN_API_KEY`

## Next Steps

Ready to create your custom multicall contract for aggregating vault data efficiently!