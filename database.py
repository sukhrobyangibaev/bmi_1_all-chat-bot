from datetime import datetime
import sqlite3
from contextlib import contextmanager
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class SQLiteDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_tables()
        logger.info(f"Database initialized with path={db_path}")

    def _create_tables(self):
        """Create the necessary tables if they don't exist."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create users table (unchanged)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    preferred_provider TEXT,
                    preferred_model TEXT
                )
                """)
                
                # Create conversations table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    title TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
                """)
                
                # Create messages table with conversation support
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    conversation_id INTEGER,
                    provider TEXT,
                    model TEXT,
                    content TEXT,
                    role TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                )
                """)
                
                conn.commit()
                logger.info("Tables created successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {str(e)}")
            raise

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_user(self, user_id: int) -> Optional[dict]:
        logger.info(f"Fetching user data for user_id: {user_id}")
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                result = cur.fetchone()
                if result:
                    logger.info(f"User {user_id} found in database.")
                    return dict(result)
                else:
                    logger.info(f"User {user_id} not found in database.")
                    return None
        except sqlite3.Error as e:
            logger.error(f"Error fetching user {user_id}: {str(e)}")
            raise

    def create_user(self, user_id: int, username: str, first_name: str, last_name: str = None):
        logger.info(f"Creating/updating user - user_id: {user_id}, username: {username}")
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO users (user_id, username, first_name, last_name)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        username=excluded.username,
                        first_name=excluded.first_name,
                        last_name=excluded.last_name
                """, (user_id, username, first_name, last_name))
                conn.commit()
                result = cur.lastrowid
                logger.info(f"User {user_id} successfully created/updated.")
                return result
        except sqlite3.Error as e:
            logger.error(f"Error creating/updating user {user_id}: {str(e)}")
            raise

    def get_user_message_count(self, user_id: int, period_hours: int = 24) -> int:
        """Get the number of messages sent by a user in the last period_hours."""
        logger.info(f"Checking message count for user_id: {user_id}")
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT COUNT(*) FROM messages 
                    WHERE user_id = ? 
                    AND role = 'user'
                    AND created_at > datetime('now', ? || ' hours')
                """, (user_id, -period_hours))
                return cur.fetchone()[0]
        except sqlite3.Error as e:
            logger.error(f"Error checking message count for user {user_id}: {str(e)}")
            raise

    def save_message(self, user_id: int, provider: str, model: str, content: str, role: str) -> str:
        """Save a message to the database."""
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                # Get active conversation or create new one
                active_conv = self.get_active_conversation(user_id)
                if not active_conv:
                    conversation_id = self.create_conversation(user_id)
                else:
                    conversation_id = active_conv['conversation_id']
                    
                cur.execute("""
                    INSERT INTO messages (
                        user_id, conversation_id, provider, model, content, role, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """, (user_id, conversation_id, provider, model, content, role))
                return cur.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error saving message for user {user_id}: {str(e)}")
            raise

    def get_messages_by_user(self, user_id: int) -> list[dict]:
        """Get all messages for the active conversation."""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                active_conv = self.get_active_conversation(user_id)
                if not active_conv:
                    return []
                    
                cur.execute("""
                    SELECT role, content, provider, model, created_at
                    FROM messages 
                    WHERE user_id = ? AND conversation_id = ?
                    ORDER BY created_at DESC
                """, (user_id, active_conv['conversation_id']))
                
                messages = [dict(row) for row in cur.fetchall()]
                messages.reverse()
                return messages
        except sqlite3.Error as e:
            logger.error(f"Error fetching messages for user {user_id}: {str(e)}")
            raise

    def get_user_ai_preferences(self, user_id: int) -> Optional[dict]:
        """Get user's AI provider and model preferences."""
        logger.info(f"Fetching AI preferences for user_id: {user_id}")
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("""
                    SELECT preferred_provider, preferred_model
                    FROM users 
                    WHERE user_id = ?
                """, (user_id,))
                result = cur.fetchone()
                if result:
                    preferences = {
                        'provider': result['preferred_provider'],
                        'model': result['preferred_model']
                    }
                    logger.info(f"Found AI preferences for user {user_id}: {preferences}")
                    return preferences
                logger.info(f"No AI preferences found for user {user_id}")
                return None
        except sqlite3.Error as e:
            logger.error(f"Error fetching AI preferences for user {user_id}: {str(e)}")
            raise

    def update_user_ai_preferences(self, user_id: int, provider: str, model: str):
        """Update user's AI provider and model preferences."""
        logger.info(f"Updating AI preferences for user_id: {user_id} - provider: {provider}, model: {model}")
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    UPDATE users 
                    SET preferred_provider = ?, preferred_model = ?
                    WHERE user_id = ?
                """, (provider, model, user_id))
                logger.info(f"Successfully updated AI preferences for user {user_id}")
        except sqlite3.Error as e:
            logger.error(f"Error updating AI preferences for user {user_id}: {str(e)}")
            raise

    def get_user_stats(self, user_id: int) -> dict:
        """Get comprehensive statistics for a user."""
        logger.info(f"Fetching stats for user_id: {user_id}")
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                stats = {}

                # Total messages
                cur.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) as user_messages,
                           SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) as bot_messages
                    FROM messages 
                    WHERE user_id = ?
                """, (user_id,))
                result = cur.fetchone()
                stats['total_messages'] = result[0]
                stats['user_messages'] = result[1]
                stats['bot_messages'] = result[2]

                # Messages by provider
                cur.execute("""
                    SELECT provider, COUNT(*) as count
                    FROM messages 
                    WHERE user_id = ? AND role = 'assistant'
                    GROUP BY provider
                """, (user_id,))
                stats['messages_by_provider'] = dict(cur.fetchall())

                # Most used model
                cur.execute("""
                    SELECT model, COUNT(*) as count
                    FROM messages 
                    WHERE user_id = ? AND role = 'assistant'
                    GROUP BY model
                    ORDER BY count DESC
                    LIMIT 1
                """, (user_id,))
                result = cur.fetchone()
                stats['most_used_model'] = result[0] if result else None

                # Time-based statistics
                periods = {
                    'today': "datetime('now', 'start of day')",
                    'this_week': "datetime('now', '-6 days')",
                    'this_month': "datetime('now', '-29 days')"
                }
                
                stats['time_periods'] = {}
                for period_name, period_start in periods.items():
                    cur.execute(f"""
                        SELECT COUNT(*) 
                        FROM messages 
                        WHERE user_id = ? 
                        AND created_at >= {period_start}
                    """, (user_id,))
                    stats['time_periods'][period_name] = cur.fetchone()[0]

                return stats
        except sqlite3.Error as e:
            logger.error(f"Error fetching stats for user {user_id}: {str(e)}")
            raise

    def create_conversation(self, user_id: int, title: str = None) -> int:
        """Create a new conversation and return its ID."""
        try:
            with self.get_connection() as conn:
                # Deactivate all other conversations for this user
                cur = conn.cursor()
                cur.execute("""
                    UPDATE conversations 
                    SET is_active = 0 
                    WHERE user_id = ?
                """, (user_id,))
                
                # Create new conversation
                if not title:
                    title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                
                cur.execute("""
                    INSERT INTO conversations (user_id, title, is_active)
                    VALUES (?, ?, 1)
                """, (user_id, title))
                
                return cur.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error creating conversation for user {user_id}: {str(e)}")
            raise

    def get_conversation(self, conversation_id: int) -> Optional[dict]:
        """Get conversation details by ID."""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("""
                    SELECT * FROM conversations 
                    WHERE conversation_id = ?
                """, (conversation_id,))
                result = cur.fetchone()
                return dict(result) if result else None
        except sqlite3.Error as e:
            logger.error(f"Error fetching conversation {conversation_id}: {str(e)}")
            raise

    def get_user_conversations(self, user_id: int, limit: int = None, offset: int = 0) -> list[dict]:
        """Get all conversations for a user with pagination support."""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                query = """
                    SELECT c.*, COUNT(m.id) as message_count
                    FROM conversations c
                    LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                    WHERE c.user_id = ?
                    GROUP BY c.conversation_id
                    ORDER BY c.created_at DESC
                """
                params = [user_id]
                
                if limit is not None:
                    query += " LIMIT ? OFFSET ?"
                    params.extend([limit, offset])
                
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Error fetching conversations for user {user_id}: {str(e)}")
            raise

    def update_conversation_title(self, conversation_id: int, title: str):
        """Update conversation title."""
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    UPDATE conversations 
                    SET title = ?
                    WHERE conversation_id = ?
                """, (title, conversation_id))
        except sqlite3.Error as e:
            logger.error(f"Error updating conversation {conversation_id} title: {str(e)}")
            raise

    def set_conversation_active(self, conversation_id: int, user_id: int):
        """Set a conversation as active and deactivate others."""
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                # Deactivate all conversations for this user
                cur.execute("""
                    UPDATE conversations 
                    SET is_active = 0 
                    WHERE user_id = ?
                """, (user_id,))
                # Activate the selected conversation
                cur.execute("""
                    UPDATE conversations 
                    SET is_active = 1
                    WHERE conversation_id = ?
                """, (conversation_id,))
        except sqlite3.Error as e:
            logger.error(f"Error setting conversation {conversation_id} active: {str(e)}")
            raise

    def get_active_conversation(self, user_id: int) -> Optional[dict]:
        """Get the active conversation for a user."""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("""
                    SELECT * FROM conversations 
                    WHERE user_id = ? AND is_active = 1
                    LIMIT 1
                """, (user_id,))
                result = cur.fetchone()
                return dict(result) if result else None
        except sqlite3.Error as e:
            logger.error(f"Error fetching active conversation for user {user_id}: {str(e)}")
            raise