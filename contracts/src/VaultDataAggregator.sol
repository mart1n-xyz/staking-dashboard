// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

/// @notice Interface for StakeManager contract
interface IStakeManager {
    struct VaultData {
        uint256 stakedBalance;
        uint256 rewardIndex;
        uint256 mpAccrued;
        uint256 maxMP;
        uint256 lastMPUpdateTime;
        uint256 rewardsAccrued;
    }

    function getVault(address vaultAddress) external view returns (VaultData memory);
    function vaultOwners(address vault) external view returns (address owner);
}

/// @notice Interface for StakeVault contract
interface IStakeVault {
    function lockUntil() external view returns (uint256);
}

/**
 * @title VaultDataAggregator
 * @notice Efficiently aggregates vault data from StakeManager and individual vaults in a single call
 * @dev Reduces RPC calls by batching multiple contract reads into one transaction
 */
contract VaultDataAggregator {
    /// @notice Aggregated vault data structure returned by the contract
    struct AggregatedVaultData {
        address vaultAddress;      // The vault contract address
        address owner;             // Current vault owner
        uint256 lockUntil;         // Timestamp when vault unlocks
        uint256 stakedBalance;     // Amount of tokens staked in the vault
        uint256 mpAccrued;         // Multiplier points accrued
        uint256 lastMPUpdateTime;  // Last time MP was updated
        uint256 rewardsAccrued;    // Rewards accrued in the vault
        bool success;              // Whether all calls succeeded for this vault
    }

    /**
     * @notice Aggregates vault data from multiple vaults in a single call
     * @param stakeManager Address of the StakeManager contract
     * @param vaults Array of vault addresses to query
     * @return vaultData Array of aggregated vault data
     */
    function getVaultsData(
        address stakeManager,
        address[] calldata vaults
    ) external view returns (AggregatedVaultData[] memory vaultData) {
        vaultData = new AggregatedVaultData[](vaults.length);
        
        IStakeManager sm = IStakeManager(stakeManager);
        
        for (uint256 i = 0; i < vaults.length; i++) {
            address vaultAddress = vaults[i];
            vaultData[i].vaultAddress = vaultAddress;
            
            bool allCallsSucceeded = true;
            
            try sm.getVault(vaultAddress) returns (IStakeManager.VaultData memory smVaultData) {
                vaultData[i].stakedBalance = smVaultData.stakedBalance;
                vaultData[i].mpAccrued = smVaultData.mpAccrued;
                vaultData[i].lastMPUpdateTime = smVaultData.lastMPUpdateTime;
                vaultData[i].rewardsAccrued = smVaultData.rewardsAccrued;
            } catch {
                allCallsSucceeded = false;
            }
            
            try sm.vaultOwners(vaultAddress) returns (address owner) {
                vaultData[i].owner = owner;
            } catch {
                allCallsSucceeded = false;
            }
            
            try IStakeVault(vaultAddress).lockUntil() returns (uint256 lockUntil) {
                vaultData[i].lockUntil = lockUntil;
            } catch {
                allCallsSucceeded = false;
            }
            
            vaultData[i].success = allCallsSucceeded;
        }
        
        return vaultData;
    }

    /**
     * @notice Get aggregated data for a single vault
     * @param stakeManager Address of the StakeManager contract
     * @param vault Address of the vault to query
     * @return vaultData Aggregated vault data
     */
    function getVaultData(
        address stakeManager,
        address vault
    ) external view returns (AggregatedVaultData memory vaultData) {
        address[] memory vaults = new address[](1);
        vaults[0] = vault;
        
        AggregatedVaultData[] memory results = this.getVaultsData(stakeManager, vaults);
        return results[0];
    }

    /**
     * @notice Get the number of vaults that can be processed in a single call
     * @dev This is a rough estimate based on gas limits. Actual limit may vary.
     * @return maxVaults Estimated maximum number of vaults per call
     */
    function getMaxVaultsPerCall() external pure returns (uint256 maxVaults) {
        // Conservative estimate: ~50k gas per vault, target ~10M gas
        return 200;
    }
}
