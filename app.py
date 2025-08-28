import streamlit as st
import requests
from web3 import Web3
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json

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

# Get RPC endpoint from environment
rpc_endpoint = os.getenv("RPC_ENDPOINT")

# Retrieve data button
if st.button("🔍 Retrieve Data", type="primary"):
    if not rpc_endpoint:
        st.error("Please set RPC_ENDPOINT in your .env file")
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

# Vault addresses section
st.subheader("📋 Discovered Staking Vaults")

# Display vault data if available
if 'vault_df' in st.session_state and not st.session_state.vault_df.empty:
    vault_df = st.session_state.vault_df
    
    # Handle backward compatibility: rename old column if it exists
    if 'owner_address' in vault_df.columns and 'deployer_address' not in vault_df.columns:
        vault_df = vault_df.rename(columns={'owner_address': 'deployer_address'})
        st.session_state.vault_df = vault_df  # Update the stored version
    
    # Display summary with data science stats
    st.write(f"**Total Vaults Found**: {len(vault_df)}")
    
    # Use the correct column name for deployers
    deployer_col = 'deployer_address' if 'deployer_address' in vault_df.columns else 'owner_address'
    st.write(f"**Unique Deployers**: {vault_df[deployer_col].nunique()}")
    
    # Display vault table (reset index to show vault_address as column)
    display_df = vault_df.reset_index()
    
    # Select only the columns we want to show (hide technical columns)
    display_columns = ['vault_address', deployer_col, 'block_number', 'transaction_hash']
    available_columns = [col for col in display_columns if col in display_df.columns]
    display_df = display_df[available_columns]
    
    # Set up column config
    column_config = {
        "vault_address": st.column_config.TextColumn("Vault Address", width="large"),
        "block_number": st.column_config.NumberColumn("Block Number", width="small"),
        "transaction_hash": st.column_config.TextColumn("Transaction Hash", width="large")
    }
    
    # Add deployer column config with the correct name
    if deployer_col in display_df.columns:
        column_config[deployer_col] = st.column_config.TextColumn("Deployer Address", width="large")
    
    st.dataframe(
        display_df,
        column_config=column_config,
        hide_index=True,
        width="stretch"
    )
    
else:
    st.info("Click 'Retrieve Data' to discover staking vaults from the blockchain.")
