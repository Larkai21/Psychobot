-- Migration 003: Row-Level Security Policies and Access Control
-- 
-- This migration implements comprehensive security for the Psychobot therapeutic system:
-- 1. Creates database roles (patient, psychologist, admin)
-- 2. Creates assignments table for patient-psychologist mapping
-- 3. Implements RLS policies for all therapeutic tables
-- 4. Adds encryption functions for sensitive clinical data
-- 5. Creates audit logging for access control

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- SECTION 1: DATABASE ROLES
-- ============================================================================

-- Function to create therapeutic roles
CREATE OR REPLACE FUNCTION create_therapeutic_roles()
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    -- Create roles if they don't exist
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'patient') THEN
        CREATE ROLE patient;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'psychologist') THEN
        CREATE ROLE psychologist;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'admin') THEN
        CREATE ROLE admin;
    END IF;
    
    -- Grant basic permissions
    GRANT USAGE ON SCHEMA public TO patient, psychologist, admin;
    GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO patient;
    GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO psychologist;
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin;
    
    RETURN TRUE;
EXCEPTION
    WHEN OTHERS THEN
        RAISE LOG 'Error creating therapeutic roles: %', SQLERRM;
        RETURN FALSE;
END;
$$;

-- ============================================================================
-- SECTION 2: ASSIGNMENTS TABLE
-- ============================================================================

-- Create assignments table for patient-psychologist mapping
CREATE TABLE IF NOT EXISTS assignments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id TEXT NOT NULL,
    psychologist_id TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    active BOOLEAN DEFAULT TRUE,
    notes TEXT,
    
    -- Ensure unique active assignments
    CONSTRAINT unique_active_assignment UNIQUE (patient_id, psychologist_id, active)
);

-- Create indexes for assignments table
CREATE INDEX IF NOT EXISTS idx_assignments_patient_id ON assignments(patient_id);
CREATE INDEX IF NOT EXISTS idx_assignments_psychologist_id ON assignments(psychologist_id);
CREATE INDEX IF NOT EXISTS idx_assignments_active ON assignments(active) WHERE active = TRUE;

-- Add trigger for updated_at
CREATE OR REPLACE FUNCTION update_assignments_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trigger_assignments_updated_at ON assignments;
CREATE TRIGGER trigger_assignments_updated_at
    BEFORE UPDATE ON assignments
    FOR EACH ROW
    EXECUTE FUNCTION update_assignments_updated_at();

-- ============================================================================
-- SECTION 3: USER ROLES TABLE
-- ============================================================================

-- Create user_roles table to track user role assignments
CREATE TABLE IF NOT EXISTS user_roles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL CHECK (role IN ('patient', 'psychologist', 'admin')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    active BOOLEAN DEFAULT TRUE
);

-- Create indexes for user_roles table
CREATE INDEX IF NOT EXISTS idx_user_roles_user_id ON user_roles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_role ON user_roles(role);

-- Add trigger for updated_at
CREATE OR REPLACE FUNCTION update_user_roles_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trigger_user_roles_updated_at ON user_roles;
CREATE TRIGGER trigger_user_roles_updated_at
    BEFORE UPDATE ON user_roles
    FOR EACH ROW
    EXECUTE FUNCTION update_user_roles_updated_at();

-- ============================================================================
-- SECTION 4: AUDIT LOG TABLE
-- ============================================================================

-- Create audit_log table for access tracking
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    target_patient_id TEXT,
    operation TEXT NOT NULL,
    table_name TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for audit_log table
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_target_patient ON audit_log(target_patient_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_success ON audit_log(success);

-- ============================================================================
-- SECTION 5: ENCRYPTION FUNCTIONS
-- ============================================================================

-- Function to encrypt clinical data
CREATE OR REPLACE FUNCTION encrypt_clinical_data(p_data TEXT, p_key TEXT)
RETURNS TABLE(encrypted_data TEXT)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT encode(pgp_sym_encrypt(p_data, p_key), 'base64') AS encrypted_data;
EXCEPTION
    WHEN OTHERS THEN
        RAISE LOG 'Error encrypting clinical data: %', SQLERRM;
        RETURN QUERY SELECT NULL::TEXT AS encrypted_data;
END;
$$;

-- Function to decrypt clinical data
CREATE OR REPLACE FUNCTION decrypt_clinical_data(p_encrypted_data TEXT, p_key TEXT)
RETURNS TABLE(decrypted_data TEXT)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT pgp_sym_decrypt(decode(p_encrypted_data, 'base64'), p_key) AS decrypted_data;
EXCEPTION
    WHEN OTHERS THEN
        RAISE LOG 'Error decrypting clinical data: %', SQLERRM;
        RETURN QUERY SELECT NULL::TEXT AS decrypted_data;
END;
$$;

-- ============================================================================
-- SECTION 6: HELPER FUNCTIONS
-- ============================================================================

-- Function to set user role
CREATE OR REPLACE FUNCTION set_user_role(p_user_id TEXT, p_role TEXT)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    INSERT INTO user_roles (user_id, role)
    VALUES (p_user_id, p_role)
    ON CONFLICT (user_id) 
    DO UPDATE SET 
        role = EXCLUDED.role,
        updated_at = NOW(),
        active = TRUE;
    
    RETURN TRUE;
EXCEPTION
    WHEN OTHERS THEN
        RAISE LOG 'Error setting user role: %', SQLERRM;
        RETURN FALSE;
END;
$$;

-- Function to get user role
CREATE OR REPLACE FUNCTION get_user_role(p_user_id TEXT)
RETURNS TABLE(role TEXT)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT ur.role
    FROM user_roles ur
    WHERE ur.user_id = p_user_id AND ur.active = TRUE;
END;
$$;

-- Function to assign patient to psychologist
CREATE OR REPLACE FUNCTION assign_patient_to_psychologist(p_patient_id TEXT, p_psychologist_id TEXT)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    INSERT INTO assignments (patient_id, psychologist_id)
    VALUES (p_patient_id, p_psychologist_id)
    ON CONFLICT (patient_id, psychologist_id, active)
    DO UPDATE SET 
        updated_at = NOW(),
        active = TRUE;
    
    RETURN TRUE;
EXCEPTION
    WHEN OTHERS THEN
        RAISE LOG 'Error assigning patient to psychologist: %', SQLERRM;
        RETURN FALSE;
END;
$$;

-- Function to remove patient assignment
CREATE OR REPLACE FUNCTION remove_patient_assignment(p_patient_id TEXT, p_psychologist_id TEXT)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    UPDATE assignments
    SET active = FALSE, updated_at = NOW()
    WHERE patient_id = p_patient_id 
    AND psychologist_id = p_psychologist_id 
    AND active = TRUE;
    
    RETURN TRUE;
EXCEPTION
    WHEN OTHERS THEN
        RAISE LOG 'Error removing patient assignment: %', SQLERRM;
        RETURN FALSE;
END;
$$;

-- Function to get psychologist's patients
CREATE OR REPLACE FUNCTION get_psychologist_patients(p_psychologist_id TEXT)
RETURNS TABLE(patient_id TEXT)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT a.patient_id
    FROM assignments a
    WHERE a.psychologist_id = p_psychologist_id AND a.active = TRUE;
END;
$$;

-- Function to audit access attempts
CREATE OR REPLACE FUNCTION audit_access_attempt(
    p_user_id TEXT,
    p_target_patient_id TEXT,
    p_operation TEXT,
    p_table_name TEXT,
    p_success BOOLEAN
)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    INSERT INTO audit_log (user_id, target_patient_id, operation, table_name, success)
    VALUES (p_user_id, p_target_patient_id, p_operation, p_table_name, p_success);
    
    RETURN TRUE;
EXCEPTION
    WHEN OTHERS THEN
        RAISE LOG 'Error creating audit log: %', SQLERRM;
        RETURN FALSE;
END;
$$;

-- ============================================================================
-- SECTION 7: ROW-LEVEL SECURITY POLICIES
-- ============================================================================

-- Function to apply all RLS policies
CREATE OR REPLACE FUNCTION apply_therapeutic_rls_policies()
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    -- Enable RLS on all therapeutic tables
    ALTER TABLE patients ENABLE ROW LEVEL SECURITY;
    ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
    ALTER TABLE session_chunks ENABLE ROW LEVEL SECURITY;
    ALTER TABLE short_term_memory ENABLE ROW LEVEL SECURITY;
    ALTER TABLE medium_term_memory ENABLE ROW LEVEL SECURITY;
    ALTER TABLE long_term_memory ENABLE ROW LEVEL SECURITY;
    ALTER TABLE assignments ENABLE ROW LEVEL SECURITY;
    ALTER TABLE user_roles ENABLE ROW LEVEL SECURITY;
    ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;

    -- Drop existing policies if they exist
    DROP POLICY IF EXISTS patients_patient_select ON patients;
    DROP POLICY IF EXISTS patients_patient_update ON patients;
    DROP POLICY IF EXISTS patients_psychologist_select ON patients;
    DROP POLICY IF EXISTS patients_admin_all ON patients;

    DROP POLICY IF EXISTS sessions_patient_select ON sessions;
    DROP POLICY IF EXISTS sessions_patient_insert ON sessions;
    DROP POLICY IF EXISTS sessions_psychologist_select ON sessions;
    DROP POLICY IF EXISTS sessions_psychologist_update ON sessions;
    DROP POLICY IF EXISTS sessions_admin_all ON sessions;

    DROP POLICY IF EXISTS chunks_patient_select ON session_chunks;
    DROP POLICY IF EXISTS chunks_patient_insert ON session_chunks;
    DROP POLICY IF EXISTS chunks_psychologist_select ON session_chunks;
    DROP POLICY IF EXISTS chunks_admin_all ON session_chunks;

    DROP POLICY IF EXISTS stm_patient_select ON short_term_memory;
    DROP POLICY IF EXISTS stm_patient_insert ON short_term_memory;
    DROP POLICY IF EXISTS stm_psychologist_select ON short_term_memory;
    DROP POLICY IF EXISTS stm_admin_all ON short_term_memory;

    DROP POLICY IF EXISTS mtm_patient_select ON medium_term_memory;
    DROP POLICY IF EXISTS mtm_psychologist_select ON medium_term_memory;
    DROP POLICY IF EXISTS mtm_psychologist_insert ON medium_term_memory;
    DROP POLICY IF EXISTS mtm_admin_all ON medium_term_memory;

    DROP POLICY IF EXISTS ltm_patient_select ON long_term_memory;
    DROP POLICY IF EXISTS ltm_psychologist_select ON long_term_memory;
    DROP POLICY IF EXISTS ltm_psychologist_insert ON long_term_memory;
    DROP POLICY IF EXISTS ltm_admin_all ON long_term_memory;

    DROP POLICY IF EXISTS assignments_psychologist_select ON assignments;
    DROP POLICY IF EXISTS assignments_admin_all ON assignments;

    DROP POLICY IF EXISTS user_roles_admin_all ON user_roles;
    DROP POLICY IF EXISTS audit_log_admin_select ON audit_log;

    -- PATIENTS TABLE POLICIES
    CREATE POLICY patients_patient_select ON patients
        FOR SELECT TO patient
        USING (auth.uid()::text = patient_id);

    CREATE POLICY patients_patient_update ON patients
        FOR UPDATE TO patient
        USING (auth.uid()::text = patient_id);

    CREATE POLICY patients_psychologist_select ON patients
        FOR SELECT TO psychologist
        USING (EXISTS (
            SELECT 1 FROM assignments 
            WHERE patient_id = patients.patient_id 
            AND psychologist_id = auth.uid()::text 
            AND active = TRUE
        ));

    CREATE POLICY patients_admin_all ON patients
        FOR ALL TO admin
        USING (true);

    -- SESSIONS TABLE POLICIES
    CREATE POLICY sessions_patient_select ON sessions
        FOR SELECT TO patient
        USING (auth.uid()::text = patient_id);

    CREATE POLICY sessions_patient_insert ON sessions
        FOR INSERT TO patient
        WITH CHECK (auth.uid()::text = patient_id);

    CREATE POLICY sessions_psychologist_select ON sessions
        FOR SELECT TO psychologist
        USING (EXISTS (
            SELECT 1 FROM assignments 
            WHERE patient_id = sessions.patient_id 
            AND psychologist_id = auth.uid()::text 
            AND active = TRUE
        ));

    CREATE POLICY sessions_psychologist_update ON sessions
        FOR UPDATE TO psychologist
        USING (EXISTS (
            SELECT 1 FROM assignments 
            WHERE patient_id = sessions.patient_id 
            AND psychologist_id = auth.uid()::text 
            AND active = TRUE
        ));

    CREATE POLICY sessions_admin_all ON sessions
        FOR ALL TO admin
        USING (true);

    -- SESSION_CHUNKS TABLE POLICIES
    CREATE POLICY chunks_patient_select ON session_chunks
        FOR SELECT TO patient
        USING (auth.uid()::text = patient_id);

    CREATE POLICY chunks_patient_insert ON session_chunks
        FOR INSERT TO patient
        WITH CHECK (auth.uid()::text = patient_id);

    CREATE POLICY chunks_psychologist_select ON session_chunks
        FOR SELECT TO psychologist
        USING (EXISTS (
            SELECT 1 FROM assignments 
            WHERE patient_id = session_chunks.patient_id 
            AND psychologist_id = auth.uid()::text 
            AND active = TRUE
        ));

    CREATE POLICY chunks_admin_all ON session_chunks
        FOR ALL TO admin
        USING (true);

    -- SHORT_TERM_MEMORY TABLE POLICIES
    CREATE POLICY stm_patient_select ON short_term_memory
        FOR SELECT TO patient
        USING (auth.uid()::text = patient_id);

    CREATE POLICY stm_patient_insert ON short_term_memory
        FOR INSERT TO patient
        WITH CHECK (auth.uid()::text = patient_id);

    CREATE POLICY stm_psychologist_select ON short_term_memory
        FOR SELECT TO psychologist
        USING (EXISTS (
            SELECT 1 FROM assignments 
            WHERE patient_id = short_term_memory.patient_id 
            AND psychologist_id = auth.uid()::text 
            AND active = TRUE
        ));

    CREATE POLICY stm_admin_all ON short_term_memory
        FOR ALL TO admin
        USING (true);

    -- MEDIUM_TERM_MEMORY TABLE POLICIES
    CREATE POLICY mtm_patient_select ON medium_term_memory
        FOR SELECT TO patient
        USING (auth.uid()::text = patient_id);

    CREATE POLICY mtm_psychologist_select ON medium_term_memory
        FOR SELECT TO psychologist
        USING (EXISTS (
            SELECT 1 FROM assignments 
            WHERE patient_id = medium_term_memory.patient_id 
            AND psychologist_id = auth.uid()::text 
            AND active = TRUE
        ));

    CREATE POLICY mtm_psychologist_insert ON medium_term_memory
        FOR INSERT TO psychologist
        WITH CHECK (EXISTS (
            SELECT 1 FROM assignments 
            WHERE patient_id = medium_term_memory.patient_id 
            AND psychologist_id = auth.uid()::text 
            AND active = TRUE
        ));

    CREATE POLICY mtm_admin_all ON medium_term_memory
        FOR ALL TO admin
        USING (true);

    -- LONG_TERM_MEMORY TABLE POLICIES
    CREATE POLICY ltm_patient_select ON long_term_memory
        FOR SELECT TO patient
        USING (auth.uid()::text = patient_id);

    CREATE POLICY ltm_psychologist_select ON long_term_memory
        FOR SELECT TO psychologist
        USING (EXISTS (
            SELECT 1 FROM assignments 
            WHERE patient_id = long_term_memory.patient_id 
            AND psychologist_id = auth.uid()::text 
            AND active = TRUE
        ));

    CREATE POLICY ltm_psychologist_insert ON long_term_memory
        FOR INSERT TO psychologist
        WITH CHECK (EXISTS (
            SELECT 1 FROM assignments 
            WHERE patient_id = long_term_memory.patient_id 
            AND psychologist_id = auth.uid()::text 
            AND active = TRUE
        ));

    CREATE POLICY ltm_admin_all ON long_term_memory
        FOR ALL TO admin
        USING (true);

    -- ASSIGNMENTS TABLE POLICIES
    CREATE POLICY assignments_psychologist_select ON assignments
        FOR SELECT TO psychologist
        USING (auth.uid()::text = psychologist_id);

    CREATE POLICY assignments_admin_all ON assignments
        FOR ALL TO admin
        USING (true);

    -- USER_ROLES TABLE POLICIES
    CREATE POLICY user_roles_admin_all ON user_roles
        FOR ALL TO admin
        USING (true);

    -- AUDIT_LOG TABLE POLICIES
    CREATE POLICY audit_log_admin_select ON audit_log
        FOR SELECT TO admin
        USING (true);

    RETURN TRUE;
EXCEPTION
    WHEN OTHERS THEN
        RAISE LOG 'Error applying RLS policies: %', SQLERRM;
        RETURN FALSE;
END;
$$;

-- Function to create encryption functions (wrapper for organization)
CREATE OR REPLACE FUNCTION create_encryption_functions()
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    -- Encryption functions are already created above
    -- This is a placeholder for any additional encryption setup
    RETURN TRUE;
END;
$$;

-- ============================================================================
-- SECTION 8: SECURE STORAGE FUNCTIONS FOR SENSITIVE DATA
-- ============================================================================

-- Function to securely store session chunk with encryption
CREATE OR REPLACE FUNCTION store_encrypted_session_chunk(
    p_patient_id TEXT,
    p_session_id TEXT,
    p_content TEXT,
    p_embedding vector(384),
    p_metadata JSONB,
    p_encryption_key TEXT
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    chunk_id UUID;
    encrypted_content TEXT;
BEGIN
    -- Encrypt sensitive content
    SELECT encrypted_data INTO encrypted_content
    FROM encrypt_clinical_data(p_content, p_encryption_key);
    
    -- Insert encrypted chunk
    INSERT INTO session_chunks (
        patient_id, 
        session_id, 
        content, 
        embedding, 
        metadata
    )
    VALUES (
        p_patient_id,
        p_session_id,
        encrypted_content,
        p_embedding,
        p_metadata
    )
    RETURNING id INTO chunk_id;
    
    RETURN chunk_id;
EXCEPTION
    WHEN OTHERS THEN
        RAISE LOG 'Error storing encrypted session chunk: %', SQLERRM;
        RETURN NULL;
END;
$$;

-- Function to retrieve and decrypt session chunk
CREATE OR REPLACE FUNCTION retrieve_decrypted_session_chunk(
    p_chunk_id UUID,
    p_encryption_key TEXT
)
RETURNS TABLE(
    id UUID,
    patient_id TEXT,
    session_id TEXT,
    content TEXT,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        sc.id,
        sc.patient_id,
        sc.session_id,
        (SELECT decrypted_data FROM decrypt_clinical_data(sc.content, p_encryption_key)) AS content,
        sc.embedding,
        sc.metadata,
        sc.created_at
    FROM session_chunks sc
    WHERE sc.id = p_chunk_id;
END;
$$;

-- ============================================================================
-- SECTION 9: INITIALIZATION
-- ============================================================================

-- Initialize roles and policies
SELECT create_therapeutic_roles();
SELECT apply_therapeutic_rls_policies();

-- Grant execute permissions on functions
GRANT EXECUTE ON FUNCTION encrypt_clinical_data(TEXT, TEXT) TO patient, psychologist, admin;
GRANT EXECUTE ON FUNCTION decrypt_clinical_data(TEXT, TEXT) TO patient, psychologist, admin;
GRANT EXECUTE ON FUNCTION set_user_role(TEXT, TEXT) TO admin;
GRANT EXECUTE ON FUNCTION get_user_role(TEXT) TO patient, psychologist, admin;
GRANT EXECUTE ON FUNCTION assign_patient_to_psychologist(TEXT, TEXT) TO admin, psychologist;
GRANT EXECUTE ON FUNCTION remove_patient_assignment(TEXT, TEXT) TO admin, psychologist;
GRANT EXECUTE ON FUNCTION get_psychologist_patients(TEXT) TO psychologist, admin;
GRANT EXECUTE ON FUNCTION audit_access_attempt(TEXT, TEXT, TEXT, TEXT, BOOLEAN) TO patient, psychologist, admin;
GRANT EXECUTE ON FUNCTION store_encrypted_session_chunk(TEXT, TEXT, TEXT, vector(384), JSONB, TEXT) TO patient, psychologist;
GRANT EXECUTE ON FUNCTION retrieve_decrypted_session_chunk(UUID, TEXT) TO patient, psychologist, admin;

-- Create default admin user (replace with actual admin user ID)
-- INSERT INTO user_roles (user_id, role) VALUES ('admin-user-id', 'admin') ON CONFLICT DO NOTHING;

COMMIT;
