"""
Unit tests for database security policies and access control.

Tests cover:
1. Patient data isolation - patients can only access their own data
2. Psychologist assignment-based access - psychologists can only access assigned patients
3. Admin unrestricted access - admins can access all data
4. Encryption/decryption of sensitive clinical data
5. Role-based access control validation
6. Audit logging for access attempts
"""

import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from typing import Dict, List, Any

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from psy_supabase.db.policies import DatabasePolicyManager, UserRole, PolicyConfig
from psy_supabase.core.database import DatabaseManager


class TestDatabasePolicyManager(unittest.TestCase):
    """Test the database policy manager functionality."""

    def setUp(self):
        self.mock_supabase = MagicMock()
        self.policy_manager = DatabasePolicyManager(self.mock_supabase)
        
        # Mock user IDs for testing
        self.patient_id = "patient_123"
        self.psychologist_id = "psychologist_456"
        self.admin_id = "admin_789"
        self.other_patient_id = "patient_999"

    def test_initialize_roles_and_policies(self):
        """Test initialization of database roles and policies."""
        # Mock successful responses
        self.mock_supabase.rpc.return_value.execute.return_value.data = True
        
        result = self.policy_manager.initialize_roles_and_policies()
        
        self.assertTrue(result)
        self.assertTrue(self.policy_manager.policies_initialized)
        
        # Verify all initialization functions were called
        expected_calls = [
            "create_therapeutic_roles",
            "apply_therapeutic_rls_policies", 
            "create_encryption_functions"
        ]
        
        actual_calls = [call[0][0] for call in self.mock_supabase.rpc.call_args_list]
        for expected_call in expected_calls:
            self.assertIn(expected_call, actual_calls)

    def test_set_user_role(self):
        """Test setting user roles."""
        self.mock_supabase.rpc.return_value.execute.return_value.data = True
        
        result = self.policy_manager.set_user_role(self.patient_id, UserRole.PATIENT)
        
        self.assertTrue(result)
        self.mock_supabase.rpc.assert_called_with(
            "set_user_role",
            {
                "p_user_id": self.patient_id,
                "p_role": "patient"
            }
        )

    def test_assign_patient_to_psychologist(self):
        """Test patient-psychologist assignment."""
        self.mock_supabase.rpc.return_value.execute.return_value.data = True
        
        result = self.policy_manager.assign_patient_to_psychologist(
            self.patient_id, self.psychologist_id
        )
        
        self.assertTrue(result)
        self.mock_supabase.rpc.assert_called_with(
            "assign_patient_to_psychologist",
            {
                "p_patient_id": self.patient_id,
                "p_psychologist_id": self.psychologist_id
            }
        )

    def test_get_user_role(self):
        """Test retrieving user roles."""
        self.mock_supabase.rpc.return_value.execute.return_value.data = [
            {"role": "patient"}
        ]
        
        role = self.policy_manager.get_user_role(self.patient_id)
        
        self.assertEqual(role, UserRole.PATIENT)
        self.mock_supabase.rpc.assert_called_with(
            "get_user_role",
            {"p_user_id": self.patient_id}
        )

    def test_get_psychologist_patients(self):
        """Test retrieving psychologist's assigned patients."""
        self.mock_supabase.rpc.return_value.execute.return_value.data = [
            {"patient_id": self.patient_id},
            {"patient_id": "patient_456"}
        ]
        
        patients = self.policy_manager.get_psychologist_patients(self.psychologist_id)
        
        self.assertEqual(len(patients), 2)
        self.assertIn(self.patient_id, patients)
        self.assertIn("patient_456", patients)


class TestAccessControlValidation(unittest.TestCase):
    """Test access control validation logic."""

    def setUp(self):
        self.mock_supabase = MagicMock()
        self.policy_manager = DatabasePolicyManager(self.mock_supabase)
        
        self.patient_id = "patient_123"
        self.psychologist_id = "psychologist_456"
        self.admin_id = "admin_789"
        self.other_patient_id = "patient_999"

    def test_patient_access_own_data(self):
        """Test that patients can access their own data."""
        # Mock patient role
        self.mock_supabase.rpc.return_value.execute.return_value.data = [
            {"role": "patient"}
        ]
        
        # Patient accessing their own data
        result = self.policy_manager.validate_access(
            self.patient_id, self.patient_id, "SELECT"
        )
        
        self.assertTrue(result)

    def test_patient_cannot_access_other_data(self):
        """Test that patients cannot access other patients' data."""
        # Mock patient role
        self.mock_supabase.rpc.return_value.execute.return_value.data = [
            {"role": "patient"}
        ]
        
        # Patient trying to access another patient's data
        result = self.policy_manager.validate_access(
            self.patient_id, self.other_patient_id, "SELECT"
        )
        
        self.assertFalse(result)

    def test_psychologist_access_assigned_patient(self):
        """Test that psychologists can access assigned patients' data."""
        # Mock psychologist role and assigned patients
        self.mock_supabase.rpc.side_effect = [
            Mock(execute=Mock(return_value=Mock(data=[{"role": "psychologist"}]))),
            Mock(execute=Mock(return_value=Mock(data=[{"patient_id": self.patient_id}])))
        ]
        
        result = self.policy_manager.validate_access(
            self.psychologist_id, self.patient_id, "SELECT"
        )
        
        self.assertTrue(result)

    def test_psychologist_cannot_access_unassigned_patient(self):
        """Test that psychologists cannot access unassigned patients' data."""
        # Mock psychologist role and empty assigned patients
        self.mock_supabase.rpc.side_effect = [
            Mock(execute=Mock(return_value=Mock(data=[{"role": "psychologist"}]))),
            Mock(execute=Mock(return_value=Mock(data=[])))
        ]
        
        result = self.policy_manager.validate_access(
            self.psychologist_id, self.other_patient_id, "SELECT"
        )
        
        self.assertFalse(result)

    def test_admin_unrestricted_access(self):
        """Test that admins have unrestricted access."""
        # Mock admin role
        self.mock_supabase.rpc.return_value.execute.return_value.data = [
            {"role": "admin"}
        ]
        
        # Admin accessing any patient's data
        result = self.policy_manager.validate_access(
            self.admin_id, self.patient_id, "SELECT"
        )
        
        self.assertTrue(result)

    def test_no_role_access_denied(self):
        """Test that users without roles are denied access."""
        # Mock no role found
        self.mock_supabase.rpc.return_value.execute.return_value.data = []
        
        result = self.policy_manager.validate_access(
            "unknown_user", self.patient_id, "SELECT"
        )
        
        self.assertFalse(result)


class TestEncryptionFunctionality(unittest.TestCase):
    """Test encryption and decryption of sensitive data."""

    def setUp(self):
        self.mock_supabase = MagicMock()
        self.policy_manager = DatabasePolicyManager(self.mock_supabase)
        
        self.test_data = "Sensitive therapeutic content about anxiety and depression"
        self.encryption_key = "test_encryption_key_123"
        self.encrypted_data = "base64_encrypted_data_here"

    def test_encrypt_sensitive_data(self):
        """Test encryption of sensitive clinical data."""
        self.mock_supabase.rpc.return_value.execute.return_value.data = [
            {"encrypted_data": self.encrypted_data}
        ]
        
        result = self.policy_manager.encrypt_sensitive_data(
            self.test_data, self.encryption_key
        )
        
        self.assertEqual(result, self.encrypted_data)
        self.mock_supabase.rpc.assert_called_with(
            "encrypt_clinical_data",
            {
                "p_data": self.test_data,
                "p_key": self.encryption_key
            }
        )

    def test_decrypt_sensitive_data(self):
        """Test decryption of sensitive clinical data."""
        self.mock_supabase.rpc.return_value.execute.return_value.data = [
            {"decrypted_data": self.test_data}
        ]
        
        result = self.policy_manager.decrypt_sensitive_data(
            self.encrypted_data, self.encryption_key
        )
        
        self.assertEqual(result, self.test_data)
        self.mock_supabase.rpc.assert_called_with(
            "decrypt_clinical_data",
            {
                "p_encrypted_data": self.encrypted_data,
                "p_key": self.encryption_key
            }
        )

    def test_encryption_error_handling(self):
        """Test error handling in encryption operations."""
        # Mock encryption failure
        self.mock_supabase.rpc.return_value.execute.side_effect = Exception("Encryption failed")
        
        result = self.policy_manager.encrypt_sensitive_data(
            self.test_data, self.encryption_key
        )
        
        self.assertIsNone(result)

    def test_decryption_error_handling(self):
        """Test error handling in decryption operations."""
        # Mock decryption failure
        self.mock_supabase.rpc.return_value.execute.side_effect = Exception("Decryption failed")
        
        result = self.policy_manager.decrypt_sensitive_data(
            self.encrypted_data, self.encryption_key
        )
        
        self.assertIsNone(result)


class TestDatabaseManagerIntegration(unittest.TestCase):
    """Test integration of security policies with DatabaseManager."""

    def setUp(self):
        self.mock_supabase = MagicMock()
        
        # Create DatabaseManager with mocked Supabase
        with patch('psy_supabase.core.database.create_client', return_value=self.mock_supabase):
            self.db_manager = DatabaseManager(
                supabase_url="test_url",
                supabase_key="test_key",
                user_id="test_user"
            )
        
        self.patient_id = "patient_123"
        self.session_id = "session_456"
        self.encryption_key = "test_key_789"

    def test_set_encryption_key(self):
        """Test setting encryption key."""
        self.db_manager.set_encryption_key(self.encryption_key)
        self.assertEqual(self.db_manager._encryption_key, self.encryption_key)

    def test_store_encrypted_session_chunk_access_denied(self):
        """Test storing encrypted chunk with access denied."""
        # Mock access validation failure
        with patch.object(self.db_manager, 'validate_access', return_value=False):
            self.db_manager.set_encryption_key(self.encryption_key)
            
            result = self.db_manager.store_encrypted_session_chunk(
                patient_id=self.patient_id,
                session_id=self.session_id,
                content="Test content",
                embedding=[0.1] * 384,
                metadata={"emotion": "anxiety"}
            )
            
            self.assertIsNone(result)

    def test_store_encrypted_session_chunk_success(self):
        """Test successful storage of encrypted chunk."""
        # Mock access validation success and RPC response
        with patch.object(self.db_manager, 'validate_access', return_value=True):
            self.mock_supabase.rpc.return_value.execute.return_value.data = "chunk_id_123"
            self.db_manager.set_encryption_key(self.encryption_key)
            
            result = self.db_manager.store_encrypted_session_chunk(
                patient_id=self.patient_id,
                session_id=self.session_id,
                content="Test content",
                embedding=[0.1] * 384,
                metadata={"emotion": "anxiety"}
            )
            
            self.assertEqual(result, "chunk_id_123")

    def test_retrieve_decrypted_chunk_access_denied(self):
        """Test retrieving decrypted chunk with access denied."""
        # Mock access validation failure
        with patch.object(self.db_manager, 'validate_access', return_value=False):
            self.db_manager.set_encryption_key(self.encryption_key)
            
            result = self.db_manager.retrieve_decrypted_session_chunk(
                chunk_id="chunk_123",
                patient_id=self.patient_id
            )
            
            self.assertIsNone(result)

    def test_retrieve_decrypted_chunk_success(self):
        """Test successful retrieval of decrypted chunk."""
        # Mock access validation success and RPC response
        with patch.object(self.db_manager, 'validate_access', return_value=True):
            mock_chunk_data = {
                "id": "chunk_123",
                "patient_id": self.patient_id,
                "session_id": self.session_id,
                "content": "Decrypted content",
                "embedding": [0.1] * 384,
                "metadata": {"emotion": "anxiety"},
                "created_at": datetime.now().isoformat()
            }
            self.mock_supabase.rpc.return_value.execute.return_value.data = [mock_chunk_data]
            self.db_manager.set_encryption_key(self.encryption_key)
            
            result = self.db_manager.retrieve_decrypted_session_chunk(
                chunk_id="chunk_123",
                patient_id=self.patient_id
            )
            
            self.assertEqual(result, mock_chunk_data)

    def test_missing_encryption_key(self):
        """Test operations without encryption key."""
        # Don't set encryption key
        result = self.db_manager.store_encrypted_session_chunk(
            patient_id=self.patient_id,
            session_id=self.session_id,
            content="Test content",
            embedding=[0.1] * 384
        )
        
        self.assertIsNone(result)


class TestAuditLogging(unittest.TestCase):
    """Test audit logging functionality."""

    def setUp(self):
        self.mock_supabase = MagicMock()
        self.policy_manager = DatabasePolicyManager(self.mock_supabase)
        
        self.user_id = "user_123"
        self.patient_id = "patient_456"

    def test_audit_access_attempt_success(self):
        """Test logging successful access attempts."""
        self.mock_supabase.rpc.return_value.execute.return_value.data = True
        
        result = self.policy_manager.audit_access_attempt(
            user_id=self.user_id,
            target_patient_id=self.patient_id,
            operation="SELECT",
            table_name="sessions",
            success=True
        )
        
        self.assertTrue(result)
        self.mock_supabase.rpc.assert_called_with(
            "audit_access_attempt",
            {
                "p_user_id": self.user_id,
                "p_target_patient_id": self.patient_id,
                "p_operation": "SELECT",
                "p_table_name": "sessions",
                "p_success": True
            }
        )

    def test_audit_access_attempt_failure(self):
        """Test logging failed access attempts."""
        self.mock_supabase.rpc.return_value.execute.return_value.data = True
        
        result = self.policy_manager.audit_access_attempt(
            user_id=self.user_id,
            target_patient_id=self.patient_id,
            operation="SELECT",
            table_name="sessions",
            success=False
        )
        
        self.assertTrue(result)


class TestPolicyDefinitions(unittest.TestCase):
    """Test policy configuration definitions."""

    def test_get_policy_definitions(self):
        """Test that policy definitions are properly structured."""
        from psy_supabase.db.policies import get_policy_definitions
        
        policies = get_policy_definitions()
        
        # Verify we have policies for all expected tables
        expected_tables = [
            "patients", "sessions", "session_chunks", 
            "short_term_memory", "medium_term_memory", "long_term_memory",
            "assignments"
        ]
        
        policy_tables = {policy.table_name for policy in policies}
        for table in expected_tables:
            self.assertIn(table, policy_tables)

    def test_policy_config_structure(self):
        """Test PolicyConfig structure."""
        from psy_supabase.db.policies import get_policy_definitions
        
        policies = get_policy_definitions()
        
        for policy in policies:
            self.assertIsInstance(policy.table_name, str)
            self.assertIsInstance(policy.policy_name, str)
            self.assertIsInstance(policy.role, UserRole)
            self.assertIsInstance(policy.operation, str)
            self.assertIsInstance(policy.condition, str)
            
            # Verify operations are valid
            valid_operations = ["SELECT", "INSERT", "UPDATE", "DELETE", "ALL"]
            self.assertIn(policy.operation, valid_operations)


class TestRoleBasedAccess(unittest.TestCase):
    """Test role-based access scenarios."""

    def setUp(self):
        self.mock_supabase = MagicMock()
        self.policy_manager = DatabasePolicyManager(self.mock_supabase)

    def test_patient_role_restrictions(self):
        """Test patient role access restrictions."""
        # Mock patient role
        self.mock_supabase.rpc.return_value.execute.return_value.data = [
            {"role": "patient"}
        ]
        
        patient_id = "patient_123"
        
        # Patient can access own data
        self.assertTrue(
            self.policy_manager.validate_access(patient_id, patient_id, "SELECT")
        )
        
        # Patient cannot access other patient's data
        self.assertFalse(
            self.policy_manager.validate_access(patient_id, "other_patient", "SELECT")
        )

    def test_psychologist_role_restrictions(self):
        """Test psychologist role access restrictions."""
        psychologist_id = "psychologist_123"
        assigned_patient = "patient_456"
        unassigned_patient = "patient_789"
        
        # Mock psychologist role and assigned patients
        self.mock_supabase.rpc.side_effect = [
            Mock(execute=Mock(return_value=Mock(data=[{"role": "psychologist"}]))),
            Mock(execute=Mock(return_value=Mock(data=[{"patient_id": assigned_patient}])))
        ]
        
        # Psychologist can access assigned patient
        self.assertTrue(
            self.policy_manager.validate_access(psychologist_id, assigned_patient, "SELECT")
        )
        
        # Reset mock for unassigned patient test
        self.mock_supabase.rpc.side_effect = [
            Mock(execute=Mock(return_value=Mock(data=[{"role": "psychologist"}]))),
            Mock(execute=Mock(return_value=Mock(data=[])))
        ]
        
        # Psychologist cannot access unassigned patient
        self.assertFalse(
            self.policy_manager.validate_access(psychologist_id, unassigned_patient, "SELECT")
        )

    def test_admin_role_permissions(self):
        """Test admin role unrestricted permissions."""
        # Mock admin role
        self.mock_supabase.rpc.return_value.execute.return_value.data = [
            {"role": "admin"}
        ]
        
        admin_id = "admin_123"
        any_patient = "patient_456"
        
        # Admin can access any patient's data
        self.assertTrue(
            self.policy_manager.validate_access(admin_id, any_patient, "SELECT")
        )
        self.assertTrue(
            self.policy_manager.validate_access(admin_id, any_patient, "UPDATE")
        )
        self.assertTrue(
            self.policy_manager.validate_access(admin_id, any_patient, "DELETE")
        )


if __name__ == '__main__':
    unittest.main()
