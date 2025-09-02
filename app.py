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
                                    for log in logs:
                                        vault_records.append({
                                            'vault_address': log.args.vault,
                                            'deployer_address': log.args.owner, 
                                            'block_number': log.blockNumber,
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
            st.metric("Active Vaults", f"{active_vaults}/{len(successful_vaults)}", 
                     help="Active vaults are those with a staked balance greater than 0 SNT")
        
        # Data table
        st.subheader("🔍 Detailed Vault Data")
        st.caption("⏰ All timestamps are displayed in UTC timezone")
        
        # Prepare display dataframe
        display_vault_df = successful_vaults[[
            'vault_address', 'owner', 'staked_balance_eth', 'mp_accrued_eth', 
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
