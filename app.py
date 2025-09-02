import streamlit as st
import requests
from web3 import Web3
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Load ABIs from files
def load_abi(filename):
    try:
        with open(f"abis/{filename}", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"ABI file {filename} not found")
        return []

STAKE_MANAGER_ABI = load_abi("stake_manager.json")
STAKE_VAULT_ABI = load_abi("stake_vault.json")
VAULT_DATA_AGGREGATOR_ABI = load_abi("vault_data_aggregator.json")

# Contract addresses
VAULT_DATA_AGGREGATOR_ADDRESS = "0xad7AF86563Ab6055e64f1f42bddEa1b7F3c0C7cA"

# Rate limiting configuration
MAX_REQUESTS_PER_SECOND = 10
MAX_REQUESTS_PER_DAY = 100000
BATCH_SIZE = 50  # Max vaults per call

class RateLimiter:
    def __init__(self, max_per_second=MAX_REQUESTS_PER_SECOND, max_per_day=MAX_REQUESTS_PER_DAY):
        self.max_per_second = max_per_second
        self.max_per_day = max_per_day
        self.requests_today = 0
        self.last_request_time = 0
        self.day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    def can_make_request(self):
        now = datetime.now()
        
        # Reset daily counter if it's a new day
        if now >= self.day_start + timedelta(days=1):
            self.requests_today = 0
            self.day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Check daily limit
        if self.requests_today >= self.max_per_day:
            return False, f"Daily limit of {self.max_per_day} requests reached"
        
        # Check per-second limit
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < (1.0 / self.max_per_second):
            sleep_time = (1.0 / self.max_per_second) - time_since_last
            return False, f"Rate limit: wait {sleep_time:.2f} seconds"
        
        return True, ""
    
    def make_request(self):
        self.last_request_time = time.time()
        self.requests_today += 1

# Initialize rate limiter
if 'rate_limiter' not in st.session_state:
    st.session_state.rate_limiter = RateLimiter()

# Page configuration
st.set_page_config(
    page_title="SNT Staking Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"  # This hides the sidebar
)

# Hide sidebar completely using CSS
st.markdown(
    """
    <style>
    .css-1d391kg {
        display: none;
    }
    .css-1y0tads {
        margin-left: 0px;
    }
    .css-1rs6os {
        margin-left: 0px;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main header
st.title("🔥 SNT Staking Dashboard")

# Contract Configuration
st.subheader("⚙️ Contract Configuration")

# Input field for StakeManagerTransparentProxy address
contract_address = st.text_input(
    "StakeManagerTransparentProxy Address:",
    value="0xa5a82CCfE29d7f384E9A072991a1F6182C28e575",
    help="Enter the address of the StakeManagerTransparentProxy contract"
)

# Get RPC endpoint from Streamlit secrets (fallback to environment for backward compatibility)
try:
    rpc_endpoint = st.secrets["default"]["RPC_ENDPOINT"]
except (KeyError, FileNotFoundError):
    # Fallback to environment variable for local development without secrets.toml
    rpc_endpoint = os.getenv("RPC_ENDPOINT")

# Retrieve data button
if st.button("🔍 Retrieve Vault Data", type="primary"):
    if not rpc_endpoint:
        st.error("Please set RPC_ENDPOINT in your .streamlit/secrets.toml file or .env file")
    elif not contract_address:
        st.error("Please provide a contract address")
    else:
        try:
            with st.spinner("Detecting contract deployment block..."):
                # Initialize Web3
                w3 = Web3(Web3.HTTPProvider(rpc_endpoint))
                
                # Add POA middleware to handle Proof of Authority chains
                try:
                    from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware
                    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                except ImportError:
                    st.warning("POA middleware not available. May have issues with POA chains.")
                
                if not w3.is_connected():
                    st.error("Failed to connect to RPC endpoint")
                else:
                    # Find deployment block using binary search
                    latest_block = w3.eth.block_number
                    
                    def has_code_at_block(block_num):
                        try:
                            code = w3.eth.get_code(contract_address, block_identifier=block_num)
                            return len(code) > 0
                        except:
                            return False
                    
                    # Binary search to find deployment block
                    left, right = 1, latest_block
                    deployment_block = None
                    
                    while left <= right:
                        mid = (left + right) // 2
                        if has_code_at_block(mid):
                            deployment_block = mid
                            right = mid - 1
                        else:
                            left = mid + 1
                    
                    if deployment_block:
                        # Now retrieve VaultRegistered logs
                        with st.spinner("Retrieving VaultRegistered logs..."):
                            try:
                                # Create contract instance
                                contract = w3.eth.contract(
                                    address=Web3.to_checksum_address(contract_address),
                                    abi=STAKE_MANAGER_ABI
                                )
                                
                                # Get VaultRegistered event logs using get_logs
                                logs = contract.events.VaultRegistered.get_logs(
                                    from_block=deployment_block,
                                    to_block='latest'
                                )
                                
                                if logs:
                                    st.success(f"🎉 Found {len(logs)} registered vault(s)!")
                                    
                                    # Convert logs to pandas DataFrame for data science workflows
                                    vault_records = []
                                    
                                    # Get unique block numbers to fetch timestamps efficiently
                                    unique_blocks = list(set(log.blockNumber for log in logs))
                                    block_timestamps = {}
                                    
                                    # Estimate deployment timestamps using smart sampling
                                    if len(unique_blocks) > 0:
                                        # Sort blocks to process chronologically
                                        sorted_blocks = sorted(unique_blocks)
                                        
                                        with st.spinner(f"Estimating deployment days for {len(unique_blocks)} blocks using smart sampling..."):
                                            
                                            api_call_count = [0]  # Use list to avoid nonlocal issues
                                            
                                            def fetch_block_timestamp(block_num):
                                                """Helper to fetch a single block timestamp via API"""
                                                try:
                                                    api_url = f"https://sepoliascan.status.network/api/v2/blocks/{block_num}"
                                                    response = requests.get(api_url, timeout=10)
                                                    api_call_count[0] += 1
                                                    if response.status_code == 200:
                                                        block_data = response.json()
                                                        timestamp_str = block_data.get('timestamp')
                                                        if timestamp_str:
                                                            timestamp_dt = pd.to_datetime(timestamp_str)
                                                            return int(timestamp_dt.timestamp())
                                                    return None
                                                except:
                                                    return None
                                            
                                            # Start with first and second blocks to establish block time
                                            first_block = sorted_blocks[0]
                                            first_timestamp = fetch_block_timestamp(first_block)
                                            
                                            if first_timestamp is None:
                                                st.warning("Could not fetch first block timestamp, skipping estimation")
                                                for block in sorted_blocks:
                                                    block_timestamps[block] = 0
                                            else:
                                                block_timestamps[first_block] = first_timestamp
                                                
                                                # If only one block, we're done
                                                if len(sorted_blocks) == 1:
                                                    pass
                                                else:
                                                    # Fetch second block to calculate average block time
                                                    second_block = sorted_blocks[1]
                                                    second_timestamp = fetch_block_timestamp(second_block)
                                                    
                                                    if second_timestamp is None:
                                                        # Fallback: assume 2 second block time (common for many chains)
                                                        avg_block_time = 2
                                                    else:
                                                        block_timestamps[second_block] = second_timestamp
                                                        avg_block_time = (second_timestamp - first_timestamp) / (second_block - first_block)
                                                    
                                                    # Now estimate remaining blocks using smart sampling
                                                    sampled_blocks = [first_block]
                                                    if len(sorted_blocks) > 1:
                                                        sampled_blocks.append(second_block)
                                                    
                                                    # Process remaining blocks with day-boundary estimation
                                                    for i in range(2 if len(sorted_blocks) > 1 else 1, len(sorted_blocks)):
                                                        current_block = sorted_blocks[i]
                                                        
                                                        # Find the most recent sampled block
                                                        last_sampled_block = sampled_blocks[-1]
                                                        last_sampled_timestamp = block_timestamps[last_sampled_block]
                                                        
                                                        # Estimate current block timestamp
                                                        estimated_timestamp = last_sampled_timestamp + (current_block - last_sampled_block) * avg_block_time
                                                        
                                                        # Check if we've crossed a day boundary (86400 seconds)
                                                        days_since_last = (estimated_timestamp - last_sampled_timestamp) / 86400
                                                        
                                                        if days_since_last >= 1.0 or (i == len(sorted_blocks) - 1):
                                                            # Fetch actual timestamp to recalibrate
                                                            actual_timestamp = fetch_block_timestamp(current_block)
                                                            
                                                            if actual_timestamp is not None:
                                                                block_timestamps[current_block] = actual_timestamp
                                                                sampled_blocks.append(current_block)
                                                                
                                                                # Recalculate block time based on latest sample
                                                                if len(sampled_blocks) >= 2:
                                                                    recent_blocks = sampled_blocks[-2:]
                                                                    time_diff = block_timestamps[recent_blocks[1]] - block_timestamps[recent_blocks[0]]
                                                                    block_diff = recent_blocks[1] - recent_blocks[0]
                                                                    avg_block_time = time_diff / block_diff
                                                            else:
                                                                # Use estimation if API fails
                                                                block_timestamps[current_block] = int(estimated_timestamp)
                                                        else:
                                                            # Use estimation for blocks within the same day
                                                            block_timestamps[current_block] = int(estimated_timestamp)
                                                    
                                                    # Fill in any missing blocks with interpolation
                                                    for block in sorted_blocks:
                                                        if block not in block_timestamps:
                                                            # Find nearest sampled blocks for interpolation
                                                            lower_blocks = [b for b in sampled_blocks if b < block]
                                                            upper_blocks = [b for b in sampled_blocks if b > block]
                                                            
                                                            if lower_blocks and upper_blocks:
                                                                lower_block = max(lower_blocks)
                                                                upper_block = min(upper_blocks)
                                                                
                                                                # Linear interpolation
                                                                lower_ts = block_timestamps[lower_block]
                                                                upper_ts = block_timestamps[upper_block]
                                                                
                                                                ratio = (block - lower_block) / (upper_block - lower_block)
                                                                interpolated_ts = lower_ts + ratio * (upper_ts - lower_ts)
                                                                block_timestamps[block] = int(interpolated_ts)
                                                            elif lower_blocks:
                                                                # Extrapolate from last known block
                                                                lower_block = max(lower_blocks)
                                                                lower_ts = block_timestamps[lower_block]
                                                                estimated_ts = lower_ts + (block - lower_block) * avg_block_time
                                                                block_timestamps[block] = int(estimated_ts)
                                                            else:
                                                                # Fallback to first block estimation
                                                                estimated_ts = first_timestamp + (block - first_block) * avg_block_time
                                                                block_timestamps[block] = int(estimated_ts)
                                            
                                            st.success(f"✅ Estimated timestamps for {len(unique_blocks)} blocks using {api_call_count[0]} API calls")
                                    
                                    for log in logs:
                                        deployment_timestamp = block_timestamps.get(log.blockNumber, 0)
                                        vault_records.append({
                                            'vault_address': log.args.vault,
                                            'deployer_address': log.args.owner, 
                                            'block_number': log.blockNumber,
                                            'deployment_timestamp': deployment_timestamp,
                                            'deployment_time': pd.to_datetime(deployment_timestamp, unit='s') if deployment_timestamp > 0 else pd.NaT,
                                            'transaction_hash': log.transactionHash.hex(),
                                            'transaction_index': log.transactionIndex,
                                            'log_index': log.logIndex,
                                            'block_hash': log.blockHash.hex()
                                        })
                                    
                                    # Create DataFrame with proper data types
                                    vault_df = pd.DataFrame(vault_records)
                                    
                                    # Optimize data types for performance and memory
                                    vault_df['vault_address'] = vault_df['vault_address'].astype('string')
                                    vault_df['deployer_address'] = vault_df['deployer_address'].astype('string')
                                    vault_df['block_number'] = vault_df['block_number'].astype('uint64')
                                    vault_df['deployment_timestamp'] = vault_df['deployment_timestamp'].astype('uint64')
                                    # deployment_time is already datetime, no conversion needed
                                    vault_df['transaction_hash'] = vault_df['transaction_hash'].astype('string')
                                    vault_df['transaction_index'] = vault_df['transaction_index'].astype('uint32')
                                    vault_df['log_index'] = vault_df['log_index'].astype('uint32')
                                    vault_df['block_hash'] = vault_df['block_hash'].astype('string')
                                    
                                    # Set vault_address as index for fast lookups
                                    vault_df.set_index('vault_address', inplace=True)
                                    
                                    # Store DataFrame in session state
                                    st.session_state.vault_df = vault_df
                                    
                                    # Get vault addresses for detailed data retrieval
                                    vault_addresses = list(vault_df.reset_index()['vault_address'])
                                    total_vaults = len(vault_addresses)
                                    
                                    with st.spinner("Retrieving detailed vault data..."):
                                        try:
                                            # Create aggregator contract instance
                                            aggregator_contract = w3.eth.contract(
                                                address=Web3.to_checksum_address(VAULT_DATA_AGGREGATOR_ADDRESS),
                                                abi=VAULT_DATA_AGGREGATOR_ABI
                                            )
                                            
                                            # Create batches
                                            batches = [vault_addresses[i:i + BATCH_SIZE] for i in range(0, len(vault_addresses), BATCH_SIZE)]
                                            total_batches = len(batches)
                                            
                                            # Progress tracking (minimal UI)
                                            progress_bar = st.progress(0)
                                            status_text = st.empty()
                                            
                                            all_vault_data = []
                                            rate_limiter = st.session_state.rate_limiter
                                            
                                            for i, batch in enumerate(batches):
                                                # Check rate limiting
                                                can_request, message = rate_limiter.can_make_request()
                                                if not can_request:
                                                    if "wait" in message:
                                                        sleep_time = float(message.split("wait ")[1].split(" seconds")[0])
                                                        time.sleep(sleep_time)
                                                    else:
                                                        st.error(message)
                                                        break
                                                
                                                # Update progress silently
                                                progress = (i + 1) / total_batches
                                                progress_bar.progress(progress)
                                                
                                                try:
                                                    # Make the contract call
                                                    batch_data = aggregator_contract.functions.getVaultsData(
                                                        Web3.to_checksum_address(contract_address),
                                                        [Web3.to_checksum_address(addr) for addr in batch]
                                                    ).call()
                                                    
                                                    # Record the request
                                                    rate_limiter.make_request()
                                                    
                                                    # Process the returned data safely
                                                    for j, vault_data in enumerate(batch_data):
                                                        try:
                                                            all_vault_data.append({
                                                                'vault_address': vault_data[0],
                                                                'owner': vault_data[1],
                                                                'lock_until': int(vault_data[2]) if vault_data[2] is not None else 0,
                                                                'staked_balance': int(vault_data[3]) if vault_data[3] is not None else 0,
                                                                'mp_accrued': int(vault_data[4]) if vault_data[4] is not None else 0,
                                                                'last_mp_update_time': int(vault_data[5]) if vault_data[5] is not None else 0,
                                                                'rewards_accrued': int(vault_data[6]) if vault_data[6] is not None else 0,
                                                                'success': bool(vault_data[7]) if vault_data[7] is not None else False
                                                            })
                                                        except (ValueError, TypeError, IndexError) as data_error:
                                                            st.warning(f"Error processing vault {j+1} in batch {i + 1}: {str(data_error)}")
                                                            # Add a failed entry for this vault
                                                            all_vault_data.append({
                                                                'vault_address': batch[j] if j < len(batch) else 'unknown',
                                                                'owner': '0x0000000000000000000000000000000000000000',
                                                                'lock_until': 0,
                                                                'staked_balance': 0,
                                                                'mp_accrued': 0,
                                                                'last_mp_update_time': 0,
                                                                'rewards_accrued': 0,
                                                                'success': False
                                                            })
                                                    
                                                except Exception as batch_error:
                                                    st.warning(f"Error processing batch {i + 1}: {str(batch_error)}")
                                                    # Add failed entries for all vaults in this batch
                                                    for addr in batch:
                                                        all_vault_data.append({
                                                            'vault_address': addr,
                                                            'owner': '0x0000000000000000000000000000000000000000',
                                                            'lock_until': 0,
                                                            'staked_balance': 0,
                                                            'mp_accrued': 0,
                                                            'last_mp_update_time': 0,
                                                            'rewards_accrued': 0,
                                                            'success': False
                                                        })
                                                    continue
                                            
                                            if all_vault_data:
                                                # Create DataFrame with vault data
                                                vault_data_df = pd.DataFrame(all_vault_data)
                                                
                                                # Convert data types and handle Wei values safely
                                                vault_data_df['vault_address'] = vault_data_df['vault_address'].astype('string')
                                                vault_data_df['owner'] = vault_data_df['owner'].astype('string')
                                                
                                                # Handle timestamps safely - convert to datetime
                                                vault_data_df['lock_until'] = vault_data_df['lock_until'].apply(
                                                    lambda x: pd.to_datetime(x, unit='s') if x > 0 else pd.NaT
                                                )
                                                vault_data_df['last_mp_update_time'] = vault_data_df['last_mp_update_time'].apply(
                                                    lambda x: pd.to_datetime(x, unit='s') if x > 0 else pd.NaT
                                                )
                                                
                                                # Keep Wei values as object type to handle large numbers
                                                vault_data_df['staked_balance_wei'] = vault_data_df['staked_balance'].astype('object')
                                                vault_data_df['mp_accrued_wei'] = vault_data_df['mp_accrued'].astype('object')
                                                vault_data_df['rewards_accrued_wei'] = vault_data_df['rewards_accrued'].astype('object')
                                                
                                                # Convert to ETH using Python's native division to handle large numbers
                                                vault_data_df['staked_balance_eth'] = vault_data_df['staked_balance'].apply(
                                                    lambda x: float(x) / 1e18 if x is not None else 0.0
                                                )
                                                vault_data_df['mp_accrued_eth'] = vault_data_df['mp_accrued'].apply(
                                                    lambda x: float(x) / 1e18 if x is not None else 0.0
                                                )
                                                vault_data_df['rewards_accrued_eth'] = vault_data_df['rewards_accrued'].apply(
                                                    lambda x: float(x) / 1e18 if x is not None else 0.0
                                                )
                                                
                                                # Merge deployment timestamp data from vault registration logs
                                                if 'vault_df' in st.session_state and not st.session_state.vault_df.empty:
                                                    # Reset index to access vault_address as column for merge
                                                    vault_registration_df = st.session_state.vault_df.reset_index()
                                                    
                                                    # Merge deployment timestamps with vault data
                                                    vault_data_df = vault_data_df.merge(
                                                        vault_registration_df[['vault_address', 'deployment_timestamp', 'deployment_time']],
                                                        on='vault_address',
                                                        how='left'
                                                )
                                                
                                                # Store in session state
                                                st.session_state.vault_data_df = vault_data_df
                                                
                                                # Clear progress indicators
                                                progress_bar.empty()
                                                status_text.empty()
                                                
                                                st.success(f"✅ Successfully retrieved detailed data for {len(vault_data_df)} vaults!")
                                                
                                            else:
                                                # Clear progress indicators on failure
                                                progress_bar.empty()
                                                status_text.empty()
                                                st.error("No vault data could be retrieved")
                                                
                                        except Exception as vault_data_error:
                                            # Clear progress indicators on error
                                            progress_bar.empty()
                                            status_text.empty()
                                            st.error(f"Error retrieving detailed vault data: {str(vault_data_error)}")
                                    
                                else:
                                    st.warning("No VaultRegistered events found")
                                    st.session_state.vault_df = pd.DataFrame()
                                    
                            except Exception as log_error:
                                st.error(f"Error retrieving logs: {str(log_error)}")
                                st.session_state.vault_df = pd.DataFrame()
                        
                    else:
                        st.error("Could not find contract deployment block")
                        
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")

# Display Vault Data Analysis
if 'vault_data_df' in st.session_state and not st.session_state.vault_data_df.empty:
    st.subheader("📈 Vault Data Analysis")
    
    vault_data_df = st.session_state.vault_data_df
    
    # Filter out failed queries
    successful_vaults = vault_data_df[vault_data_df['success'] == True].copy()
    failed_count = len(vault_data_df) - len(successful_vaults)
    
    if failed_count > 0:
        st.warning(f"⚠️ {failed_count} vault(s) failed to retrieve data")
    
    if len(successful_vaults) > 0:
        # Aggregate data by owner
        owner_aggregated = successful_vaults.groupby('owner').agg({
            'staked_balance': 'sum',
            'mp_accrued': 'sum', 
            'rewards_accrued': 'sum',
            'vault_address': 'count',  # Count of vaults per owner
            'staked_balance_eth': 'sum'
        }).reset_index()
        
        # Convert aggregated Wei values to ETH
        owner_aggregated['total_staked_eth'] = owner_aggregated['staked_balance'].apply(lambda x: float(x) / 1e18)
        owner_aggregated['total_mp_eth'] = owner_aggregated['mp_accrued'].apply(lambda x: float(x) / 1e18)
        owner_aggregated['total_rewards_eth'] = owner_aggregated['rewards_accrued'].apply(lambda x: float(x) / 1e18)
        owner_aggregated = owner_aggregated.rename(columns={'vault_address': 'vault_count'})
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Sum in Wei first to preserve precision, then convert to ETH
            total_staked_wei = successful_vaults['staked_balance'].sum()
            total_staked = float(total_staked_wei) / 1e18
            st.metric("Total SNT Staked", f"{total_staked:,.2f} SNT")
        
        with col2:
            # Sum in Wei first to preserve precision, then convert to ETH
            total_mp_wei = successful_vaults['mp_accrued'].sum()
            total_mp = float(total_mp_wei) / 1e18
            st.metric("Total MP Accrued", f"{total_mp:,.2f}")
        
        with col3:
            # Sum in Wei first to preserve precision, then convert to ETH
            total_rewards_wei = successful_vaults['rewards_accrued'].sum()
            total_rewards = float(total_rewards_wei) / 1e18
            st.metric("Total Karma Rewards", f"{total_rewards:,.2f} Karma")
        
        with col4:
            active_vaults = len(successful_vaults[successful_vaults['staked_balance_eth'] > 0])
            participation_rate = (active_vaults / len(successful_vaults)) * 100
            st.metric("Active Participation", f"{participation_rate:.1f}%", 
                     help=f"Active vaults: {active_vaults}/{len(successful_vaults)} (vaults with >0 SNT staked)")
        
        # Advanced Analytics Section
        st.markdown("---")
        st.subheader("🏆 Staking Analytics")
        
        # Create tabs for different analytics
        tab1, tab2, tab3 = st.tabs(["👑 Leaderboards", "📊 Network Stats", "📈 Performance Metrics"])
        
        with tab1:
            st.markdown("### 👑 Top Stakers")
            st.caption("Aggregated stakes across all vaults owned by each address")
            
            # Top 50 stakers
            top_stakers = owner_aggregated.nlargest(50, 'total_staked_eth')[['owner', 'total_staked_eth', 'vault_count', 'total_mp_eth']].copy()
            top_stakers['rank'] = range(1, len(top_stakers) + 1)
            top_stakers = top_stakers[['rank', 'owner', 'total_staked_eth', 'vault_count', 'total_mp_eth']]
            
            st.dataframe(
                top_stakers,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", width="small"),
                    "owner": st.column_config.TextColumn("Owner Address", width="large"),
                    "total_staked_eth": st.column_config.NumberColumn("Total Staked (SNT)", format="%.2f"),
                    "vault_count": st.column_config.NumberColumn("Vaults", width="small"),
                    "total_mp_eth": st.column_config.NumberColumn("Total MP", format="%.2f")
                },
                hide_index=True,
                use_container_width=True,
                height=400
            )
            
            st.markdown("### 🎖️ Karma Champions")
            st.caption("Top reward earners across all owned vaults")
            
            # Top 50 karma earners
            karma_champions = owner_aggregated.nlargest(50, 'total_rewards_eth')[['owner', 'total_rewards_eth', 'vault_count', 'total_staked_eth']].copy()
            karma_champions['rank'] = range(1, len(karma_champions) + 1)
            karma_champions = karma_champions[['rank', 'owner', 'total_rewards_eth', 'vault_count', 'total_staked_eth']]
            
            st.dataframe(
                karma_champions,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", width="small"),
                    "owner": st.column_config.TextColumn("Owner Address", width="large"),
                    "total_rewards_eth": st.column_config.NumberColumn("Total Karma", format="%.2f"),
                    "vault_count": st.column_config.NumberColumn("Vaults", width="small"),
                    "total_staked_eth": st.column_config.NumberColumn("Total Staked (SNT)", format="%.2f")
                },
                hide_index=True,
                use_container_width=True,
                height=400
            )
            
            st.markdown("### 🏭 Top Vault Deployers")
            st.caption("Owners with the most vaults deployed")
            
            # Top 50 vault deployers
            vault_deployers = owner_aggregated.nlargest(50, 'vault_count')[['owner', 'vault_count', 'total_staked_eth', 'total_mp_eth']].copy()
            vault_deployers['rank'] = range(1, len(vault_deployers) + 1)
            vault_deployers = vault_deployers[['rank', 'owner', 'vault_count', 'total_staked_eth', 'total_mp_eth']]
            
            st.dataframe(
                vault_deployers,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", width="small"),
                    "owner": st.column_config.TextColumn("Owner Address", width="large"),
                    "vault_count": st.column_config.NumberColumn("Vaults", width="small"),
                    "total_staked_eth": st.column_config.NumberColumn("Total Staked (SNT)", format="%.2f"),
                    "total_mp_eth": st.column_config.NumberColumn("Total MP", format="%.2f")
                },
                hide_index=True,
                use_container_width=True,
                height=400
            )
        
        with tab2:
            st.markdown("### 📊 Network Statistics")
            
            # Owner & Vault Overview
            st.markdown("#### 👥 Owner & Vault Overview")
            
            # Network stats in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                unique_owners = len(owner_aggregated)
                avg_vaults_per_owner = successful_vaults.groupby('owner').size().mean()
                st.metric("Unique Stakers", f"{unique_owners:,}")
                st.metric("Avg Vaults/Owner", f"{avg_vaults_per_owner:.1f}")
            
            with col2:
                active_owners = len(owner_aggregated[owner_aggregated['total_staked_eth'] > 0])
                st.metric("Active Owners", f"{active_owners}/{unique_owners}", 
                         help="Owners who have at least one vault with SNT staked (balance > 0)")
                st.metric("Total Vaults", f"{len(successful_vaults):,}")
            
            with col3:
                if len(owner_aggregated) > 0:
                    avg_stake_per_owner = owner_aggregated['total_staked_eth'].mean()
                    median_stake_per_owner = owner_aggregated['total_staked_eth'].median()
                    st.metric("Avg Stake/Owner", f"{avg_stake_per_owner:,.2f} SNT")
                    st.metric("Median Stake/Owner", f"{median_stake_per_owner:,.2f} SNT")
            
            st.markdown("---")
            
            # MP Accumulation Rate
            st.markdown("#### 🔥 MP Accumulation Analysis")
            
            # Calculate current MP accumulation rate across network
            current_time = datetime.now().timestamp()
            
            # Filter vaults with recent MP updates (last 30 days)
            recent_mp_vaults = successful_vaults[
                (successful_vaults['last_mp_update_time'].notna()) & 
                (successful_vaults['last_mp_update_time'] > pd.to_datetime(current_time - (30*24*3600), unit='s'))
            ].copy()
            
            if len(recent_mp_vaults) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_active_stake = recent_mp_vaults['staked_balance_eth'].sum()
                    st.metric("Active Stake", f"{total_active_stake:,.2f} SNT")
                    st.metric("Vaults with Recent MP", f"{len(recent_mp_vaults):,}")
                
                with col2:
                    total_active_mp = recent_mp_vaults['mp_accrued_eth'].sum()
                    avg_mp_per_vault = total_active_mp / len(recent_mp_vaults) if len(recent_mp_vaults) > 0 else 0
                    st.metric("Total Active MP", f"{total_active_mp:,.2f}")
                    st.metric("Avg MP/Vault", f"{avg_mp_per_vault:,.2f}")
                
                with col3:
                    # Average and Median MP to stake ratio
                    mp_to_stake_ratios = recent_mp_vaults.apply(
                        lambda row: row['mp_accrued_eth'] / row['staked_balance_eth'] if row['staked_balance_eth'] > 0 else 0, 
                        axis=1
                    )
                    avg_mp_ratio = mp_to_stake_ratios.mean()
                    median_mp_ratio = mp_to_stake_ratios.median()
                    
                    st.metric("Avg MP to Stake Ratio", f"{avg_mp_ratio:,.4f}")
                    st.metric("Median MP to Stake Ratio", f"{median_mp_ratio:,.4f}",
                             help="Median MP-to-stake ratio across vaults, less affected by outliers than average")
            else:
                st.info("No recent MP updates found in the last 30 days")
            
            st.markdown("---")
            
            # Lock Analysis Section
            st.markdown("#### 🔒 Lock Analysis")
            
            # Calculate lock statistics
            current_time_dt = pd.to_datetime(current_time, unit='s')
            
            # Identify locked vaults (lock_until > current time)
            locked_vaults = successful_vaults[
                (successful_vaults['lock_until'].notna()) & 
                (successful_vaults['lock_until'] > current_time_dt)
            ].copy()
            
            # Calculate remaining lock days for locked vaults
            if len(locked_vaults) > 0:
                locked_vaults['days_remaining'] = (locked_vaults['lock_until'] - current_time_dt).dt.days
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Lock participation metrics
                locked_vault_count = len(locked_vaults)
                lock_participation = (locked_vault_count / len(successful_vaults)) * 100 if len(successful_vaults) > 0 else 0
                st.metric("Locked Vaults", f"{locked_vault_count}/{len(successful_vaults)}")
                st.metric("Lock Participation", f"{lock_participation:.1f}%")
            
            with col2:
                # Locked stake analysis
                if len(locked_vaults) > 0:
                    locked_stake = locked_vaults['staked_balance_eth'].sum()
                    locked_stake_pct = (locked_stake / total_staked) * 100 if total_staked > 0 else 0
                    st.metric("Locked Stake", f"{locked_stake:,.2f} SNT")
                    st.metric("% of Total Stake Locked", f"{locked_stake_pct:.1f}%")
                else:
                    st.metric("Locked Stake", "0 SNT")
                    st.metric("% of Total Stake Locked", "0%")
            
            with col3:
                # Lock duration statistics
                if len(locked_vaults) > 0:
                    avg_lock_days = locked_vaults['days_remaining'].mean()
                    median_lock_days = locked_vaults['days_remaining'].median()
                    st.metric("Avg Lock Duration", f"{avg_lock_days:,.0f} days",
                             help="Average remaining lock time for currently locked vaults")
                    st.metric("Median Lock Duration", f"{median_lock_days:,.0f} days",
                             help="Median remaining lock time, less affected by extremely long locks")
                else:
                    st.metric("Avg Lock Duration", "0 days")
                    st.metric("Median Lock Duration", "0 days")
            

            
            # Owner-level lock analysis
            if len(locked_vaults) > 0:
                st.markdown("##### 👤 Owner Lock Behavior")
                
                # Aggregate by owner for lock analysis
                owner_lock_stats = locked_vaults.groupby('owner').agg({
                    'staked_balance_eth': 'sum',
                    'vault_address': 'count',
                    'days_remaining': ['mean', 'max']
                }).round(2)
                
                # Flatten column names
                owner_lock_stats.columns = ['locked_stake', 'locked_vaults', 'avg_lock_days', 'max_lock_days']
                owner_lock_stats = owner_lock_stats.reset_index()
                
                # Merge with total owner stakes to calculate lock percentage
                owner_lock_analysis = owner_aggregated.merge(
                    owner_lock_stats, on='owner', how='left'
                ).fillna(0)
                
                # Calculate what percentage of each owner's stake is locked
                owner_lock_analysis['lock_percentage'] = (
                    owner_lock_analysis['locked_stake'] / owner_lock_analysis['total_staked_eth'] * 100
                ).fillna(0)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Owners with locks
                    owners_with_locks = len(owner_lock_analysis[owner_lock_analysis['locked_stake'] > 0])
                    owner_lock_rate = (owners_with_locks / len(owner_aggregated)) * 100 if len(owner_aggregated) > 0 else 0
                    st.metric("Owners with Locks", f"{owners_with_locks}/{len(owner_aggregated)}")
                    st.metric("Owner Lock Rate", f"{owner_lock_rate:.1f}%")
                
                with col2:
                    # Average stake locked per owner
                    avg_owner_lock_pct = owner_lock_analysis['lock_percentage'].mean()
                    median_owner_lock_pct = owner_lock_analysis['lock_percentage'].median()
                    st.metric("Avg Owner Lock %", f"{avg_owner_lock_pct:.1f}%",
                             help="Average percentage of stake that owners have locked")
                    st.metric("Median Owner Lock %", f"{median_owner_lock_pct:.1f}%")
                
                with col3:
                    # Lock commitment patterns
                    full_lock_owners = len(owner_lock_analysis[owner_lock_analysis['lock_percentage'] >= 99])
                    partial_lock_owners = len(owner_lock_analysis[
                        (owner_lock_analysis['lock_percentage'] > 0) & 
                        (owner_lock_analysis['lock_percentage'] < 99)
                    ])
                    st.metric("Full Lock Owners", f"{full_lock_owners:,}",
                             help="Owners with ≥99% of their stake locked")
                    st.metric("Partial Lock Owners", f"{partial_lock_owners:,}",
                             help="Owners with some but not all stake locked")
            else:
                st.info("No currently locked vaults found")
            
            st.markdown("---")
            
            # Vault Deployment Stats
            st.markdown("#### 🚀 Vault Deployment Stats")
            
            # Filter vaults with valid deployment times
            deployed_vaults = successful_vaults[successful_vaults['deployment_time'].notna()].copy()
            
            if len(deployed_vaults) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # First and latest deployments
                    first_deployment = deployed_vaults['deployment_time'].min()
                    latest_deployment = deployed_vaults['deployment_time'].max()
                    st.metric("First Vault Deployed", first_deployment.strftime("%Y-%m-%d"))
                    st.metric("Latest Vault Deployed", latest_deployment.strftime("%Y-%m-%d"))
                
                with col2:
                    # Deployment period analysis
                    deployment_period = (latest_deployment - first_deployment).days
                    avg_deployments_per_day = len(deployed_vaults) / max(deployment_period, 1)
                    st.metric("Deployment Period", f"{deployment_period:,} days")
                    st.metric("Avg Deployments/Day", f"{avg_deployments_per_day:.2f}")
                
                with col3:
                    # Recent deployment activity (last 7 days)
                    recent_deployments = deployed_vaults[
                        deployed_vaults['deployment_time'] > pd.to_datetime(current_time - (7*24*3600), unit='s')
                    ]
                    recent_deploy_count = len(recent_deployments)
                    st.metric("Deployed in Last 7 Days", f"{recent_deploy_count:,}")
                    
                    # Calculate 7d growth rate if we have historical data
                    if deployment_period > 7:
                        prev_7d_deployments = deployed_vaults[
                            (deployed_vaults['deployment_time'] > pd.to_datetime(current_time - (14*24*3600), unit='s')) &
                            (deployed_vaults['deployment_time'] <= pd.to_datetime(current_time - (7*24*3600), unit='s'))
                        ]
                        if len(prev_7d_deployments) > 0:
                            growth_rate = ((recent_deploy_count - len(prev_7d_deployments)) / len(prev_7d_deployments)) * 100
                            st.metric("7d Growth Rate", f"{growth_rate:+.1f}%")
                        else:
                            st.metric("7d Growth Rate", "N/A")
                    else:
                        st.metric("7d Growth Rate", "N/A")
                
                # Deployment chart for all deployment history
                st.markdown("#### 📈 Daily Vault Deployments")
                
                if len(deployed_vaults) > 0:
                    # Create daily deployment counts for all deployments
                    deployed_vaults_chart = deployed_vaults.copy()
                    deployed_vaults_chart['deployment_date'] = deployed_vaults_chart['deployment_time'].dt.date
                    daily_counts = deployed_vaults_chart.groupby('deployment_date').size().reset_index(name='vault_count')
                    
                    # Create complete date range from first to latest deployment
                    first_date = deployed_vaults_chart['deployment_date'].min()
                    latest_date = deployed_vaults_chart['deployment_date'].max()
                    
                    date_range = pd.date_range(
                        start=first_date,
                        end=latest_date,
                        freq='D'
                    )
                    
                    # Create complete dataframe with zeros for missing days
                    complete_dates = pd.DataFrame({'deployment_date': date_range.date})
                    chart_data = complete_dates.merge(daily_counts, on='deployment_date', how='left').fillna(0)
                    chart_data['vault_count'] = chart_data['vault_count'].astype(int)
                    
                    # Display the chart
                    st.bar_chart(
                        chart_data.set_index('deployment_date')['vault_count'],
                        height=300
                    )
                    
                    # Show summary
                    total_deployments = chart_data['vault_count'].sum()
                    max_day = chart_data['vault_count'].max()
                    deployment_days = len(chart_data)
                    st.caption(f"📊 Total deployments: {total_deployments:,} | Peak day: {max_day:,} vaults | Period: {deployment_days:,} days")
                else:
                    st.info("No vault deployment data available")
            else:
                st.info("No deployment timestamp data available")
        
        with tab3:
            st.markdown("### 📈 Staking Performance & Metrics")
            
            # Performance metrics
            col1, col2 = st.columns(2)
            
            with col1:
                if len(successful_vaults) > 0:
                    # Average rewards per SNT
                    avg_rewards_ratio = total_rewards / total_staked if total_staked > 0 else 0
                    st.metric("Avg Karma/SNT", f"{avg_rewards_ratio:,.4f}")
            
            with col2:
                if len(successful_vaults) > 0:
                    # Vault utilization
                    multi_vault_owners = len(owner_aggregated[owner_aggregated['vault_count'] > 1])
                    multi_vault_rate = (multi_vault_owners / len(owner_aggregated)) * 100 if len(owner_aggregated) > 0 else 0
                    st.metric("Multi-Vault Owners", f"{multi_vault_rate:.1f}%")
        
        # Data table
        st.markdown("---")
        st.subheader("🔍 Detailed Vault Data")
        st.caption("⏰ All timestamps are displayed in UTC timezone")
        
        # Prepare display dataframe
        display_vault_df = successful_vaults[[
            'vault_address', 'owner', 'deployment_time', 'staked_balance_eth', 'mp_accrued_eth', 
            'rewards_accrued_eth', 'lock_until', 'last_mp_update_time'
        ]].copy()
        
        # Round numerical values for display
        display_vault_df['staked_balance_eth'] = display_vault_df['staked_balance_eth'].round(2)
        display_vault_df['mp_accrued_eth'] = display_vault_df['mp_accrued_eth'].round(2)
        display_vault_df['rewards_accrued_eth'] = display_vault_df['rewards_accrued_eth'].round(2)
        
        # Sort by staked balance descending
        display_vault_df = display_vault_df.sort_values('staked_balance_eth', ascending=False)
        
        # Column configuration for better display
        column_config = {
            "vault_address": st.column_config.TextColumn("Vault Address", width="medium"),
            "owner": st.column_config.TextColumn("Owner", width="medium"),
            "deployment_time": st.column_config.DatetimeColumn("Deployed (UTC)", width="medium"),
            "staked_balance_eth": st.column_config.NumberColumn("Staked Balance (SNT)", format="%.2f"),
            "mp_accrued_eth": st.column_config.NumberColumn("MP Accrued", format="%.2f"),
            "rewards_accrued_eth": st.column_config.NumberColumn("Karma Rewards", format="%.2f"),
            "lock_until": st.column_config.DatetimeColumn("Lock Until (UTC)"),
            "last_mp_update_time": st.column_config.DatetimeColumn("Last MP Update (UTC)")
        }
        
        st.dataframe(
            display_vault_df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True
        )

else:
    st.info("Click 'Retrieve Vault Data' to discover and analyze staking vaults from the blockchain.")
