"""
Database Policies and Security Rules for Psychobot Clinical AI System.

This module implements comprehensive Row-Level Security (RLS) policies and role-based
access control for therapeutic data protection. It ensures strict data isolation
between patients, controlled access for psychologists, and administrative oversight.

Key Features:
- Role-based access control (patient, psychologist, admin)
- Patient data isolation with RLS policies
- Psychologist assignment-based access control
- Administrative unrestricted access
- Encryption helpers for sensitive clinical data
- Policy initialization and management functions

Security Roles:
- patient: Can only access their own therapeutic data
- psychologist: Can access data for assigned patients only
- admin: Unrestricted access for system administration

Tables Covered:
- patients: Patient profile and demographic data
- sessions: Therapeutic session records
- session_chunks: Psychological text chunks with embeddings
- short_term_memory: Session-based memory storage
- medium_term_memory: Weekly clustered summaries
- long_term_memory: Longitudinal therapeutic themes
- assignments: Patient-psychologist assignment mapping
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from supabase import Client
from psy_supabase import get_package_logger

logger = get_package_logger(__name__)


class UserRole(Enum):
    """Enumeration of user roles in the therapeutic system."""
    PATIENT = "patient"
    PSYCHOLOGIST = "psychologist"
    ADMIN = "admin"


@dataclass
class PolicyConfig:
    """Configuration for RLS policy creation."""
    table_name: str
    policy_name: str
    role: UserRole
    operation: str  # SELECT, INSERT, UPDATE, DELETE
    condition: str  # SQL condition for the policy


class DatabasePolicyManager:
    """
    Manages database security policies and role-based access control.
    
    This class provides methods to initialize database roles, create RLS policies,
    and manage access control for therapeutic data across all memory layers.
    """

    def __init__(self, supabase_client: Client):
        """
        Initialize the policy manager.
        
        Args:
            supabase_client: Authenticated Supabase client
        """
        self.supabase = supabase_client
        self.policies_initialized = False

    def initialize_roles_and_policies(self) -> bool:
        """
        Initialize all database roles and RLS policies.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            logger.info("Initializing database roles and RLS policies...")
            
            # Create roles if they don't exist
            if not self._create_database_roles():
                logger.error("Failed to create database roles")
                return False
            
            # Apply RLS policies for all tables
            if not self._apply_rls_policies():
                logger.error("Failed to apply RLS policies")
                return False
            
            # Create encryption functions
            if not self._create_encryption_functions():
                logger.error("Failed to create encryption functions")
                return False
            
            self.policies_initialized = True
            logger.info("Database roles and policies initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing roles and policies: {e}")
            return False

    def _create_database_roles(self) -> bool:
        """Create database roles for patient, psychologist, and admin."""
        try:
            response = self.supabase.rpc("create_therapeutic_roles").execute()
            return response.data is not None
        except Exception as e:
            logger.error(f"Error creating database roles: {e}")
            return False

    def _apply_rls_policies(self) -> bool:
        """Apply Row-Level Security policies to all therapeutic tables."""
        try:
            response = self.supabase.rpc("apply_therapeutic_rls_policies").execute()
            return response.data is not None
        except Exception as e:
            logger.error(f"Error applying RLS policies: {e}")
            return False

    def _create_encryption_functions(self) -> bool:
        """Create encryption/decryption functions for sensitive data."""
        try:
            response = self.supabase.rpc("create_encryption_functions").execute()
            return response.data is not None
        except Exception as e:
            logger.error(f"Error creating encryption functions: {e}")
            return False

    def set_user_role(self, user_id: str, role: UserRole) -> bool:
        """
        Set the role for a specific user.
        
        Args:
            user_id: User identifier
            role: UserRole enum value
            
        Returns:
            bool: True if role was set successfully
        """
        try:
            response = self.supabase.rpc(
                "set_user_role",
                {
                    "p_user_id": user_id,
                    "p_role": role.value
                }
            ).execute()
            
            if response.data:
                logger.info(f"Set role {role.value} for user {user_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error setting user role: {e}")
            return False

    def assign_patient_to_psychologist(
        self, 
        patient_id: str, 
        psychologist_id: str
    ) -> bool:
        """
        Create an assignment between a patient and psychologist.
        
        Args:
            patient_id: Patient identifier
            psychologist_id: Psychologist identifier
            
        Returns:
            bool: True if assignment was created successfully
        """
        try:
            response = self.supabase.rpc(
                "assign_patient_to_psychologist",
                {
                    "p_patient_id": patient_id,
                    "p_psychologist_id": psychologist_id
                }
            ).execute()
            
            if response.data:
                logger.info(f"Assigned patient {patient_id} to psychologist {psychologist_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error creating patient assignment: {e}")
            return False

    def remove_patient_assignment(
        self, 
        patient_id: str, 
        psychologist_id: str
    ) -> bool:
        """
        Remove an assignment between a patient and psychologist.
        
        Args:
            patient_id: Patient identifier
            psychologist_id: Psychologist identifier
            
        Returns:
            bool: True if assignment was removed successfully
        """
        try:
            response = self.supabase.rpc(
                "remove_patient_assignment",
                {
                    "p_patient_id": patient_id,
                    "p_psychologist_id": psychologist_id
                }
            ).execute()
            
            if response.data:
                logger.info(f"Removed assignment: patient {patient_id} from psychologist {psychologist_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing patient assignment: {e}")
            return False

    def get_user_role(self, user_id: str) -> Optional[UserRole]:
        """
        Get the role for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserRole or None if not found
        """
        try:
            response = self.supabase.rpc(
                "get_user_role",
                {"p_user_id": user_id}
            ).execute()
            
            if response.data and len(response.data) > 0:
                role_str = response.data[0].get("role")
                if role_str:
                    return UserRole(role_str)
            return None
            
        except Exception as e:
            logger.error(f"Error getting user role: {e}")
            return None

    def get_psychologist_patients(self, psychologist_id: str) -> List[str]:
        """
        Get list of patients assigned to a psychologist.
        
        Args:
            psychologist_id: Psychologist identifier
            
        Returns:
            List of patient IDs
        """
        try:
            response = self.supabase.rpc(
                "get_psychologist_patients",
                {"p_psychologist_id": psychologist_id}
            ).execute()
            
            if response.data:
                return [item["patient_id"] for item in response.data]
            return []
            
        except Exception as e:
            logger.error(f"Error getting psychologist patients: {e}")
            return []

    def validate_access(
        self, 
        user_id: str, 
        target_patient_id: str, 
        operation: str = "SELECT"
    ) -> bool:
        """
        Validate if a user has access to a patient's data.
        
        Args:
            user_id: User requesting access
            target_patient_id: Patient whose data is being accessed
            operation: Type of operation (SELECT, INSERT, UPDATE, DELETE)
            
        Returns:
            bool: True if access is allowed
        """
        try:
            user_role = self.get_user_role(user_id)
            
            if not user_role:
                logger.warning(f"No role found for user {user_id}")
                return False
            
            # Admin has unrestricted access
            if user_role == UserRole.ADMIN:
                return True
            
            # Patient can only access their own data
            if user_role == UserRole.PATIENT:
                return user_id == target_patient_id
            
            # Psychologist can access assigned patients
            if user_role == UserRole.PSYCHOLOGIST:
                assigned_patients = self.get_psychologist_patients(user_id)
                return target_patient_id in assigned_patients
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating access: {e}")
            return False

    def encrypt_sensitive_data(self, data: str, encryption_key: str) -> Optional[str]:
        """
        Encrypt sensitive clinical data using pgcrypto.
        
        Args:
            data: Plaintext data to encrypt
            encryption_key: Encryption key
            
        Returns:
            Encrypted data or None if encryption failed
        """
        try:
            response = self.supabase.rpc(
                "encrypt_clinical_data",
                {
                    "p_data": data,
                    "p_key": encryption_key
                }
            ).execute()
            
            if response.data:
                return response.data[0].get("encrypted_data")
            return None
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            return None

    def decrypt_sensitive_data(self, encrypted_data: str, encryption_key: str) -> Optional[str]:
        """
        Decrypt sensitive clinical data using pgcrypto.
        
        Args:
            encrypted_data: Encrypted data to decrypt
            encryption_key: Decryption key
            
        Returns:
            Decrypted data or None if decryption failed
        """
        try:
            response = self.supabase.rpc(
                "decrypt_clinical_data",
                {
                    "p_encrypted_data": encrypted_data,
                    "p_key": encryption_key
                }
            ).execute()
            
            if response.data:
                return response.data[0].get("decrypted_data")
            return None
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            return None

    def audit_access_attempt(
        self, 
        user_id: str, 
        target_patient_id: str, 
        operation: str,
        table_name: str,
        success: bool
    ) -> bool:
        """
        Log access attempts for audit purposes.
        
        Args:
            user_id: User attempting access
            target_patient_id: Target patient ID
            operation: Operation attempted
            table_name: Table being accessed
            success: Whether access was granted
            
        Returns:
            bool: True if audit log was created
        """
        try:
            response = self.supabase.rpc(
                "audit_access_attempt",
                {
                    "p_user_id": user_id,
                    "p_target_patient_id": target_patient_id,
                    "p_operation": operation,
                    "p_table_name": table_name,
                    "p_success": success
                }
            ).execute()
            
            return response.data is not None
            
        except Exception as e:
            logger.error(f"Error creating audit log: {e}")
            return False


def get_policy_definitions() -> List[PolicyConfig]:
    """
    Get all RLS policy definitions for therapeutic tables.
    
    Returns:
        List of PolicyConfig objects defining all security policies
    """
    policies = []
    
    # Patient table policies
    policies.extend([
        PolicyConfig("patients", "patients_patient_select", UserRole.PATIENT, "SELECT", 
                    "auth.uid()::text = patient_id"),
        PolicyConfig("patients", "patients_patient_update", UserRole.PATIENT, "UPDATE", 
                    "auth.uid()::text = patient_id"),
        PolicyConfig("patients", "patients_psychologist_select", UserRole.PSYCHOLOGIST, "SELECT",
                    "EXISTS (SELECT 1 FROM assignments WHERE patient_id = patients.patient_id AND psychologist_id = auth.uid()::text)"),
        PolicyConfig("patients", "patients_admin_all", UserRole.ADMIN, "ALL", "true"),
    ])
    
    # Sessions table policies
    policies.extend([
        PolicyConfig("sessions", "sessions_patient_select", UserRole.PATIENT, "SELECT",
                    "auth.uid()::text = patient_id"),
        PolicyConfig("sessions", "sessions_patient_insert", UserRole.PATIENT, "INSERT",
                    "auth.uid()::text = patient_id"),
        PolicyConfig("sessions", "sessions_psychologist_select", UserRole.PSYCHOLOGIST, "SELECT",
                    "EXISTS (SELECT 1 FROM assignments WHERE patient_id = sessions.patient_id AND psychologist_id = auth.uid()::text)"),
        PolicyConfig("sessions", "sessions_psychologist_update", UserRole.PSYCHOLOGIST, "UPDATE",
                    "EXISTS (SELECT 1 FROM assignments WHERE patient_id = sessions.patient_id AND psychologist_id = auth.uid()::text)"),
        PolicyConfig("sessions", "sessions_admin_all", UserRole.ADMIN, "ALL", "true"),
    ])
    
    # Session chunks table policies
    policies.extend([
        PolicyConfig("session_chunks", "chunks_patient_select", UserRole.PATIENT, "SELECT",
                    "auth.uid()::text = patient_id"),
        PolicyConfig("session_chunks", "chunks_patient_insert", UserRole.PATIENT, "INSERT",
                    "auth.uid()::text = patient_id"),
        PolicyConfig("session_chunks", "chunks_psychologist_select", UserRole.PSYCHOLOGIST, "SELECT",
                    "EXISTS (SELECT 1 FROM assignments WHERE patient_id = session_chunks.patient_id AND psychologist_id = auth.uid()::text)"),
        PolicyConfig("session_chunks", "chunks_admin_all", UserRole.ADMIN, "ALL", "true"),
    ])
    
    # Short-term memory policies
    policies.extend([
        PolicyConfig("short_term_memory", "stm_patient_select", UserRole.PATIENT, "SELECT",
                    "auth.uid()::text = patient_id"),
        PolicyConfig("short_term_memory", "stm_patient_insert", UserRole.PATIENT, "INSERT",
                    "auth.uid()::text = patient_id"),
        PolicyConfig("short_term_memory", "stm_psychologist_select", UserRole.PSYCHOLOGIST, "SELECT",
                    "EXISTS (SELECT 1 FROM assignments WHERE patient_id = short_term_memory.patient_id AND psychologist_id = auth.uid()::text)"),
        PolicyConfig("short_term_memory", "stm_admin_all", UserRole.ADMIN, "ALL", "true"),
    ])
    
    # Medium-term memory policies
    policies.extend([
        PolicyConfig("medium_term_memory", "mtm_patient_select", UserRole.PATIENT, "SELECT",
                    "auth.uid()::text = patient_id"),
        PolicyConfig("medium_term_memory", "mtm_psychologist_select", UserRole.PSYCHOLOGIST, "SELECT",
                    "EXISTS (SELECT 1 FROM assignments WHERE patient_id = medium_term_memory.patient_id AND psychologist_id = auth.uid()::text)"),
        PolicyConfig("medium_term_memory", "mtm_psychologist_insert", UserRole.PSYCHOLOGIST, "INSERT",
                    "EXISTS (SELECT 1 FROM assignments WHERE patient_id = medium_term_memory.patient_id AND psychologist_id = auth.uid()::text)"),
        PolicyConfig("medium_term_memory", "mtm_admin_all", UserRole.ADMIN, "ALL", "true"),
    ])
    
    # Long-term memory policies
    policies.extend([
        PolicyConfig("long_term_memory", "ltm_patient_select", UserRole.PATIENT, "SELECT",
                    "auth.uid()::text = patient_id"),
        PolicyConfig("long_term_memory", "ltm_psychologist_select", UserRole.PSYCHOLOGIST, "SELECT",
                    "EXISTS (SELECT 1 FROM assignments WHERE patient_id = long_term_memory.patient_id AND psychologist_id = auth.uid()::text)"),
        PolicyConfig("long_term_memory", "ltm_psychologist_insert", UserRole.PSYCHOLOGIST, "INSERT",
                    "EXISTS (SELECT 1 FROM assignments WHERE patient_id = long_term_memory.patient_id AND psychologist_id = auth.uid()::text)"),
        PolicyConfig("long_term_memory", "ltm_admin_all", UserRole.ADMIN, "ALL", "true"),
    ])
    
    # Assignments table policies (psychologists and admins only)
    policies.extend([
        PolicyConfig("assignments", "assignments_psychologist_select", UserRole.PSYCHOLOGIST, "SELECT",
                    "auth.uid()::text = psychologist_id"),
        PolicyConfig("assignments", "assignments_admin_all", UserRole.ADMIN, "ALL", "true"),
    ])
    
    return policies
