// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import {Script, console} from "forge-std/Script.sol";
import {VaultDataAggregator} from "../src/VaultDataAggregator.sol";

contract DeployScript is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        
        vm.startBroadcast(deployerPrivateKey);
        
        VaultDataAggregator aggregator = new VaultDataAggregator();
        
        console.log("VaultDataAggregator deployed at:", address(aggregator));
        console.log("Max vaults per call:", aggregator.getMaxVaultsPerCall());
        
        vm.stopBroadcast();
    }
}
