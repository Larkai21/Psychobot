-- Migration: Create session_chunks table for psychological text chunking
-- This table stores therapeutic chunks with emotional and psychological metadata

-- Create session_chunks table in default schema
CREATE TABLE IF NOT EXISTS session_chunks (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    interaction_id INTEGER,
    chunk_text TEXT NOT NULL,
    embedding vector(384), -- Default embedding dimension for sentence-transformers
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_session_chunks_session_id ON session_chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_session_chunks_interaction_id ON session_chunks(interaction_id);
CREATE INDEX IF NOT EXISTS idx_session_chunks_created_at ON session_chunks(created_at);

-- Create vector similarity index for embeddings
CREATE INDEX IF NOT EXISTS idx_session_chunks_embedding ON session_chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create GIN index for metadata queries
CREATE INDEX IF NOT EXISTS idx_session_chunks_metadata ON session_chunks USING GIN (metadata);

-- Create function to add session chunks
CREATE OR REPLACE FUNCTION add_session_chunk(
    p_schema_name TEXT,
    p_session_id TEXT,
    p_interaction_id INTEGER,
    p_chunk_text TEXT,
    p_embedding vector(384),
    p_metadata JSONB DEFAULT '{}'
) RETURNS INTEGER AS $$
DECLARE
    chunk_id INTEGER;
    table_name TEXT;
BEGIN
    -- Construct table name with schema
    table_name := p_schema_name || '.session_chunks';
    
    -- Insert chunk and return ID
    EXECUTE format('
        INSERT INTO %I (session_id, interaction_id, chunk_text, embedding, metadata)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
    ', table_name)
    USING p_session_id, p_interaction_id, p_chunk_text, p_embedding, p_metadata
    INTO chunk_id;
    
    RETURN chunk_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to get session chunks with filtering
CREATE OR REPLACE FUNCTION get_session_chunks(
    p_schema_name TEXT,
    p_session_id TEXT,
    p_emotion_filter TEXT DEFAULT NULL,
    p_theme_filter TEXT DEFAULT NULL
) RETURNS TABLE (
    id INTEGER,
    session_id TEXT,
    interaction_id INTEGER,
    chunk_text TEXT,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
DECLARE
    table_name TEXT;
    query_text TEXT;
BEGIN
    -- Construct table name with schema
    table_name := p_schema_name || '.session_chunks';
    
    -- Base query
    query_text := format('SELECT id, session_id, interaction_id, chunk_text, embedding, metadata, created_at FROM %I WHERE session_id = $1', table_name);
    
    -- Add emotion filter if provided
    IF p_emotion_filter IS NOT NULL THEN
        query_text := query_text || ' AND metadata->>''primary_emotion'' = $2';
    END IF;
    
    -- Add theme filter if provided
    IF p_theme_filter IS NOT NULL THEN
        IF p_emotion_filter IS NOT NULL THEN
            query_text := query_text || ' AND metadata->>''theme'' = $3';
        ELSE
            query_text := query_text || ' AND metadata->>''theme'' = $2';
        END IF;
    END IF;
    
    -- Add ordering
    query_text := query_text || ' ORDER BY created_at ASC';
    
    -- Execute query based on filters
    IF p_emotion_filter IS NOT NULL AND p_theme_filter IS NOT NULL THEN
        RETURN QUERY EXECUTE query_text USING p_session_id, p_emotion_filter, p_theme_filter;
    ELSIF p_emotion_filter IS NOT NULL THEN
        RETURN QUERY EXECUTE query_text USING p_session_id, p_emotion_filter;
    ELSIF p_theme_filter IS NOT NULL THEN
        RETURN QUERY EXECUTE query_text USING p_session_id, p_theme_filter;
    ELSE
        RETURN QUERY EXECUTE query_text USING p_session_id;
    END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to find similar chunks by embedding
CREATE OR REPLACE FUNCTION find_similar_chunks(
    p_schema_name TEXT,
    p_embedding vector(384),
    p_limit INTEGER DEFAULT 5,
    p_min_similarity FLOAT DEFAULT 0.1
) RETURNS TABLE (
    id INTEGER,
    session_id TEXT,
    chunk_text TEXT,
    metadata JSONB,
    similarity FLOAT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
DECLARE
    table_name TEXT;
BEGIN
    -- Construct table name with schema
    table_name := p_schema_name || '.session_chunks';
    
    RETURN QUERY EXECUTE format('
        SELECT 
            sc.id,
            sc.session_id,
            sc.chunk_text,
            sc.metadata,
            1 - (sc.embedding <=> $1) as similarity,
            sc.created_at
        FROM %I sc
        WHERE 1 - (sc.embedding <=> $1) >= $2
        ORDER BY sc.embedding <=> $1
        LIMIT $3
    ', table_name)
    USING p_embedding, p_min_similarity, p_limit;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to analyze emotional trajectory
CREATE OR REPLACE FUNCTION analyze_emotional_trajectory(
    p_schema_name TEXT,
    p_session_id TEXT
) RETURNS TABLE (
    chunk_count INTEGER,
    avg_polarity FLOAT,
    avg_intensity FLOAT,
    dominant_emotion TEXT,
    dominant_theme TEXT,
    emotion_distribution JSONB,
    theme_distribution JSONB
) AS $$
DECLARE
    table_name TEXT;
BEGIN
    -- Construct table name with schema
    table_name := p_schema_name || '.session_chunks';
    
    RETURN QUERY EXECUTE format('
        WITH chunk_stats AS (
            SELECT 
                COUNT(*) as total_chunks,
                AVG((metadata->>''polarity'')::float) as avg_pol,
                AVG((metadata->>''intensity'')::float) as avg_int,
                mode() WITHIN GROUP (ORDER BY metadata->>''primary_emotion'') as dom_emotion,
                mode() WITHIN GROUP (ORDER BY metadata->>''theme'') as dom_theme
            FROM %I
            WHERE session_id = $1
        ),
        emotion_dist AS (
            SELECT jsonb_object_agg(
                metadata->>''primary_emotion'', 
                emotion_count
            ) as emotions
            FROM (
                SELECT 
                    metadata->>''primary_emotion'' as emotion,
                    COUNT(*) as emotion_count
                FROM %I
                WHERE session_id = $1
                GROUP BY metadata->>''primary_emotion''
            ) e
        ),
        theme_dist AS (
            SELECT jsonb_object_agg(
                metadata->>''theme'', 
                theme_count
            ) as themes
            FROM (
                SELECT 
                    metadata->>''theme'' as theme,
                    COUNT(*) as theme_count
                FROM %I
                WHERE session_id = $1
                GROUP BY metadata->>''theme''
            ) t
        )
        SELECT 
            cs.total_chunks::integer,
            ROUND(cs.avg_pol::numeric, 2)::float,
            ROUND(cs.avg_int::numeric, 2)::float,
            cs.dom_emotion,
            cs.dom_theme,
            COALESCE(ed.emotions, ''{}''::jsonb),
            COALESCE(td.themes, ''{}''::jsonb)
        FROM chunk_stats cs
        CROSS JOIN emotion_dist ed
        CROSS JOIN theme_dist td
    ', table_name, table_name, table_name)
    USING p_session_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_session_chunks_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to session_chunks table
CREATE TRIGGER trigger_update_session_chunks_updated_at
    BEFORE UPDATE ON session_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_session_chunks_updated_at();

-- Grant permissions (adjust as needed for your security model)
-- GRANT SELECT, INSERT, UPDATE ON session_chunks TO authenticated;
-- GRANT USAGE ON SEQUENCE session_chunks_id_seq TO authenticated;

-- Add comments for documentation
COMMENT ON TABLE session_chunks IS 'Stores psychological text chunks with emotional and therapeutic metadata';
COMMENT ON COLUMN session_chunks.session_id IS 'Session identifier linking chunks to conversations';
COMMENT ON COLUMN session_chunks.interaction_id IS 'Optional reference to parent interaction';
COMMENT ON COLUMN session_chunks.chunk_text IS 'The actual text content of the therapeutic chunk';
COMMENT ON COLUMN session_chunks.embedding IS 'Vector embedding for semantic similarity search';
COMMENT ON COLUMN session_chunks.metadata IS 'JSONB containing emotion, polarity, intensity, theme, and other psychological metadata';

-- Create schema-specific table creation function for user schemas
CREATE OR REPLACE FUNCTION create_session_chunks_table_in_schema(schema_name TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    table_name TEXT;
BEGIN
    -- Construct full table name
    table_name := schema_name || '.session_chunks';
    
    -- Create table in the specified schema
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            interaction_id INTEGER,
            chunk_text TEXT NOT NULL,
            embedding vector(384),
            metadata JSONB DEFAULT ''{}''::jsonb,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )', table_name);
    
    -- Create indexes
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I(session_id)', 
                   'idx_' || schema_name || '_session_chunks_session_id', table_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I(interaction_id)', 
                   'idx_' || schema_name || '_session_chunks_interaction_id', table_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I(created_at)', 
                   'idx_' || schema_name || '_session_chunks_created_at', table_name);
    
    -- Create vector index
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)', 
                   'idx_' || schema_name || '_session_chunks_embedding', table_name);
    
    -- Create metadata index
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I USING GIN (metadata)', 
                   'idx_' || schema_name || '_session_chunks_metadata', table_name);
    
    -- Create trigger
    EXECUTE format('
        CREATE TRIGGER %I
            BEFORE UPDATE ON %I
            FOR EACH ROW
            EXECUTE FUNCTION update_session_chunks_updated_at()',
        'trigger_update_' || schema_name || '_session_chunks_updated_at', table_name);
    
    RETURN TRUE;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Error creating session_chunks table in schema %: %', schema_name, SQLERRM;
        RETURN FALSE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
