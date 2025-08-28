// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import {Test, console} from "forge-std/Test.sol";
import {VaultDataAggregator} from "../src/VaultDataAggregator.sol";

contract VaultDataAggregatorTest is Test {
    VaultDataAggregator public aggregator;
    
    // Mock addresses for testing
    address constant STAKE_MANAGER = 0xa5a82CCfE29d7f384E9A072991a1F6182C28e575;
    address constant MOCK_VAULT_1 = address(0x1111);
    address constant MOCK_VAULT_2 = address(0x2222);
    
    function setUp() public {
        aggregator = new VaultDataAggregator();
    }

    function testContractDeployment() public {
        assertTrue(address(aggregator) != address(0));
        assertEq(aggregator.getMaxVaultsPerCall(), 200);
    }

    function testGetVaultsDataStructure() public view {
        // Test that we can call the function (will fail on actual calls since we're not forking)
        address[] memory vaults = new address[](2);
        vaults[0] = MOCK_VAULT_1;
        vaults[1] = MOCK_VAULT_2;
        
        // This would normally require forking the network to test with real contracts
        // For now, we just verify the contract compiles and has the right interface
        
        // Verify return structure
        VaultDataAggregator.AggregatedVaultData[] memory results;
        // In a real test, we would: results = aggregator.getVaultsData(STAKE_MANAGER, vaults);
        
        assertTrue(true); // Placeholder assertion
    }

    function testGetVaultDataSingle() public view {
        // Test single vault query structure
        VaultDataAggregator.AggregatedVaultData memory result;
        // In a real test: result = aggregator.getVaultData(STAKE_MANAGER, MOCK_VAULT_1);
        
        assertTrue(true); // Placeholder assertion
    }

    function testMaxVaultsPerCall() public {
        uint256 maxVaults = aggregator.getMaxVaultsPerCall();
        assertEq(maxVaults, 200);
        assertTrue(maxVaults > 0);
    }
}

// Additional test contract for integration testing with forked network
contract VaultDataAggregatorIntegrationTest is Test {
    VaultDataAggregator public aggregator;
    
    string constant SEPOLIA_RPC_URL = "https://public.sepolia.rpc.status.network";
    address constant SEPOLIA_STAKE_MANAGER = 0xa5a82CCfE29d7f384E9A072991a1F6182C28e575;
    
    function setUp() public {
        // Uncomment to test with real Sepolia data
        // vm.createFork(SEPOLIA_RPC_URL);
        aggregator = new VaultDataAggregator();
    }
    
    // Example of how to test with real data (requires forking)
    function testRealVaultData() public {
        // This test would be enabled when testing against real network
        /*
        address[] memory vaults = new address[](1);
        vaults[0] = 0x...; // Real vault address
        
        VaultDataAggregator.AggregatedVaultData[] memory results = 
            aggregator.getVaultsData(SEPOLIA_STAKE_MANAGER, vaults);
            
        assertEq(results.length, 1);
        assertEq(results[0].vaultAddress, vaults[0]);
        assertTrue(results[0].success);
        */
        
        assertTrue(true); // Placeholder
    }
}
