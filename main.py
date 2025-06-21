import logging
from abc import ABC, abstractmethod
import openai
from anthropic import Anthropic
from typing import Optional, Tuple
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from enum import Enum
from dataclasses import dataclass
import os
from dotenv import load_dotenv
from functools import lru_cache
from database import SQLiteDatabase as Database

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Configuration ---
@dataclass
class BotConfig:
    """Bot configuration settings."""
    token: str
    webhook_url: Optional[str]
    environment: str
    debug: bool
    openai_api_key: str
    anthropic_api_key: str
    default_ai_provider: str
    db_path: str
    google_api_key: str
    groq_api_key: str
    xai_api_key: str

    @classmethod
    def from_env(cls):
        """Create configuration from environment variables."""
        load_dotenv()
        return cls(
            token=os.getenv("TELEGRAM_BOT_TOKEN"),
            webhook_url=os.getenv("WEBHOOK_URL"),
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "False").lower() == "true",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            default_ai_provider=os.getenv("DEFAULT_AI_PROVIDER", "openai"),
            db_path=os.getenv("DB_PATH", "database.sqlite"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            xai_api_key=os.getenv("XAI_API_KEY")
        )

# --- AI Providers ---
class AIProvider(ABC):
    @abstractmethod
    async def generate_response(self, messages: list[dict]) -> str:
        pass

class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        
    async def generate_response(self, messages: list[dict]) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

class AnthropicProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-latest"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        
    async def generate_response(self, messages: list[dict]) -> str:
        try:
            # Extract system prompt and convert messages
            system_prompt = next((m["content"] for m in messages if m["role"] == "system"), None)
            anthropic_messages = [
                {
                    "role": "assistant" if msg["role"] == "assistant" else "user",
                    "content": msg["content"]
                }
                for msg in messages
                if msg["role"] != "system"
            ]
            
            response = self.client.messages.create(
                model=self.model,
                messages=anthropic_messages,
                system=system_prompt,
                max_tokens=1024
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")

class GoogleProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model = model
        
    async def generate_response(self, messages: list[dict]) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Google API error: {str(e)}")
            raise

class GroqProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "groq-1.0"):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = model
        
    async def generate_response(self, messages: list[dict]) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise

class XAIProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "xai-1.0"):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        self.model = model
        
    async def generate_response(self, messages: list[dict]) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"XAI API error: {str(e)}")
            raise

class AIProviderFactory:
    _provider_instances = {}
    
    @staticmethod
    @lru_cache(maxsize=32)
    def create_provider(provider_name: str, api_key: str, model: str = None) -> Optional[AIProvider]:
        cache_key = f"{provider_name}:{model}"
        if cache_key not in AIProviderFactory._provider_instances:
            providers = {
                "openai": lambda: OpenAIProvider(api_key, model or "gpt-4o-mini"),
                "anthropic": lambda: AnthropicProvider(api_key, model or "claude-3-5-haiku-latest"),
                "google": lambda: GoogleProvider(api_key, model or "gemini-1.5-flash"),
                "groq": lambda: GroqProvider(api_key, model or "llama-3.1-8b-instant"),
                "xai": lambda: XAIProvider(api_key, model or "grok-beta")
            }
            AIProviderFactory._provider_instances[cache_key] = providers.get(provider_name.lower(), lambda: None)()
        return AIProviderFactory._provider_instances[cache_key]


# --- Bot ---
class ConversationState(Enum):
    SELECTING_PROVIDER = 1
    SELECTING_MODEL = 2

class TelegramBot:
    def __init__(self, database: Database, config: BotConfig):
        """Initialize bot with config and create application instance."""
        self.config = config
        self.database = database
        self.application = Application.builder().token(self.config.token).build()
        # Add command registry
        self.commands = {}
        # Initialize provider_models and other attributes first
        self.provider_models = {
            "openai": ["gpt-4o", "chatgpt-4o-latest", "gpt-4o-mini"],
            "anthropic": ["claude-3-5-sonnet-latest", "claude-3-opus-latest", "claude-3-5-haiku-latest"],
            "google": ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"],
            "groq": ["llama3-8b-8192", "llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
            "xai": ["grok-beta"]
        }
        # Set default provider initially
        self.current_provider = AIProviderFactory.create_provider(
            config.default_ai_provider,
            getattr(config, f"{config.default_ai_provider}_api_key")
        )
        if not self.current_provider:
            raise ValueError(f"Invalid AI provider: {config.default_ai_provider}")
        
        self.message_history = {}
        self._setup_handlers()

    def _setup_handlers(self):
        """Configure message and command handlers."""
        # Add basic commands
        self.register_command("start", self.start_command, "Start the bot")
        self.register_command("help", self.help_command, "Show help message")
        self.register_command("switch_ai", self.switch_ai_command, "Switch AI provider")
        self.register_command("new", self.new_chat_command, "Start a new conversation")
        self.register_command("stats", self.stats_command, "Show your chat statistics")
        self.register_command("conversations", self.list_conversations_command, "List your conversations")
        self.register_command("rename_conversation", self.rename_conversation_command, "Rename current conversation")
        
        # Message handler for text messages
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
        
        # Error handler
        self.application.add_error_handler(self.error_handler)
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))

    def register_command(self, command: str, handler, description: str):
        """Register a new command with its handler and description."""
        self.commands[command] = {"handler": handler, "description": description}
        self.application.add_handler(CommandHandler(command, handler))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user = update.effective_user
        # Create or update user in database
        self.database.create_user(
            user_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name
        )
        
        # Get user's preferred AI provider and model from database
        preferences = self.database.get_user_ai_preferences(user.id)
        if preferences and preferences.get('provider') and preferences.get('model'):
            api_key = getattr(self.config, f"{preferences['provider']}_api_key", None)
            if api_key:
                self.current_provider = AIProviderFactory.create_provider(
                    preferences['provider'],
                    api_key,
                    preferences['model']
                )
        
        # Get message count from history
        message_history = self.database.get_messages_by_user(user.id)
        
        welcome_message = (
            f"👋 Hi {user.first_name}!\n\n"
            "I'm your AI assistant bot. I can help you interact with various AI models.\n"
            "Use /help to see available commands.\n\n"
            f"---\n"
            f"Provider: {self.current_provider.__class__.__name__.replace('Provider', '')}\n"
            f"Model: {self.current_provider.model}\n"
            f"History: {len(message_history)} messages"
        )
        await update.message.reply_text(welcome_message)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help message."""
        # Get message count from history
        message_history = self.database.get_messages_by_user(update.effective_user.id)
        
        help_text = (
            "🤖 Available commands:\n\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/switch_ai - Switch between different AI providers\n"
            "/new - Start a fresh conversation (clears chat history)\n"
            "/stats - Show your chat statistics\n"
            "/conversations - List your conversations\n"
            "/rename_conversation - Rename current conversation\n\n"
            "Simply send me a message and I'll respond using AI!\n\n"
            f"---\n"
            f"Provider: {self.current_provider.__class__.__name__.replace('Provider', '')}\n"
            f"Model: {self.current_provider.model}\n"
            f"History: {len(message_history)} messages"
        )
        await update.message.reply_text(help_text)

    async def switch_ai_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /switch_ai command."""
        # Get message count from history
        message_history = self.database.get_messages_by_user(update.effective_user.id)
        
        status_text = (
            "Choose an AI provider:\n\n"
            f"---\n"
            f"Current Provider: {self.current_provider.__class__.__name__.replace('Provider', '')}\n"
            f"Current Model: {self.current_provider.model}\n"
            f"History: {len(message_history)} messages"
        )
        
        keyboard = []
        for provider in self.provider_models.keys():
            keyboard.append([InlineKeyboardButton(provider.title(), callback_data=f"provider_{provider}")])
        keyboard.append([InlineKeyboardButton("Cancel", callback_data="cancel")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            status_text,
            reply_markup=reply_markup
        )

    async def new_chat_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /new command to start a fresh conversation."""
        user_id = update.effective_user.id
        
        try:
            # Create new conversation
            conversation_id = self.database.create_conversation(user_id)
            active_conv = self.database.get_conversation(conversation_id)
            
            await update.message.reply_text(
                f"🔄 Started a new conversation: {active_conv['title']}\n\n"
                f"---\n"
                f"Provider: {self.current_provider.__class__.__name__.replace('Provider', '')}\n"
                f"Model: {self.current_provider.model}"
            )
        except Exception as e:
            logger.error(f"Error starting new chat for user {user_id}: {str(e)}")
            await update.message.reply_text(
                "Sorry, I encountered an error while trying to start a new chat. Please try again later."
            )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards."""
        query = update.callback_query
        await query.answer()
        
        if query.data == "cancel":
            await query.edit_message_text("Operation cancelled.")
            return
        
        # Handle conversation pagination
        if query.data in ["conv_page_next", "conv_page_prev"]:
            current_page = context.user_data.get('conv_page', 0)
            context.user_data['conv_page'] = current_page + 1 if query.data == "conv_page_next" else current_page - 1
            
            # Re-fetch conversations with new page
            per_page = 5
            page = context.user_data['conv_page']
            conversations = self.database.get_user_conversations(
                query.from_user.id,
                limit=per_page + 1,
                offset=page * per_page
            )
            
            if not conversations:
                context.user_data['conv_page'] = max(0, current_page - 1)  # Reset to previous page
                await query.edit_message_text("No more conversations.")
                return
                
            has_next = len(conversations) > per_page
            conversations = conversations[:per_page]
            
            message = "Your conversations:\n\n"
            for conv in conversations:
                active_marker = "🟢 " if conv['is_active'] else "⚪️ "
                message += f"{active_marker}{conv['title']} ({conv['message_count']} messages)\n"
                message += f"Created: {conv['created_at']}\n\n"
            
            keyboard = []
            for conv in conversations:
                keyboard.append([InlineKeyboardButton(
                    f"{'✓ ' if conv['is_active'] else ''}{conv['title']}",
                    callback_data=f"conv_{conv['conversation_id']}"
                )])
            
            nav_buttons = []
            if page > 0:
                nav_buttons.append(InlineKeyboardButton("Newer", callback_data="conv_page_prev"))
            if has_next:
                nav_buttons.append(InlineKeyboardButton("Older", callback_data="conv_page_next"))
            if nav_buttons:
                keyboard.append(nav_buttons)
            
            await query.edit_message_text(
                message,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return
        
        if query.data.startswith("conv_"):
            conversation_id = int(query.data.replace("conv_", ""))
            try:
                self.database.set_conversation_active(conversation_id, query.from_user.id)
                conv = self.database.get_conversation(conversation_id)
                await query.edit_message_text(f"Switched to conversation: {conv['title']}")
            except Exception as e:
                logger.error(f"Error switching conversation: {str(e)}")
                await query.edit_message_text("Failed to switch conversation.")
            return
            
        if query.data.startswith("provider_") or query.data == "back_to_providers":
            provider = query.data.replace("provider_", "") if query.data.startswith("provider_") else None
            keyboard = []
            if provider:
                for model in self.provider_models[provider]:
                    keyboard.append([InlineKeyboardButton(model, callback_data=f"model_{provider}_{model}")])
                keyboard.append([InlineKeyboardButton("Back", callback_data="back_to_providers")])
                await query.edit_message_text(
                    f"Choose a model for {provider.title()}:" if provider else "Choose an AI provider:",
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
            else:
                for provider in self.provider_models.keys():
                    keyboard.append([InlineKeyboardButton(provider.title(), callback_data=f"provider_{provider}")])
                keyboard.append([InlineKeyboardButton("Cancel", callback_data="cancel")])
                await query.edit_message_text(
                    "Choose an AI provider:",
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
            return
            
        if query.data.startswith("model_"):
            _, provider, model = query.data.split("_")
            api_key = getattr(self.config, f"{provider}_api_key", None)
            
            if not api_key:
                await query.edit_message_text(f"Invalid or unconfigured AI provider: {provider}")
                return
                
            new_provider = AIProviderFactory.create_provider(provider, api_key, model)
            if new_provider:
                self.current_provider = new_provider
                # Save user's preference to database
                try:
                    self.database.update_user_ai_preferences(
                        user_id=query.from_user.id,
                        provider=provider,
                        model=model
                    )
                    await query.edit_message_text(f"Successfully switched to {provider} ({model})")
                except Exception as e:
                    logger.error(f"Error saving AI preferences: {str(e)}")
                    await query.edit_message_text(
                        f"Switched to {provider} ({model}), but failed to save preference."
                    )
            else:
                await query.edit_message_text(f"Failed to switch to {provider} ({model})")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id

        # Check if we're waiting for a conversation rename
        if context.user_data.get('awaiting_rename'):
            conversation_id = context.user_data['awaiting_rename']
            new_title = update.message.text
            
            try:
                self.database.update_conversation_title(conversation_id, new_title)
                del context.user_data['awaiting_rename']
                await update.message.reply_text(f"Conversation renamed to: {new_title}")
                return
            except Exception as e:
                logger.error(f"Error renaming conversation: {str(e)}")
                await update.message.reply_text(
                    "Sorry, I encountered an error while renaming the conversation. Please try again."
                )
                return
        
        # Get or create active conversation
        active_conv = self.database.get_active_conversation(user_id)
        if not active_conv:
            conversation_id = self.database.create_conversation(user_id)
        else:
            conversation_id = active_conv['conversation_id']
        
        # Retrieve all conversation history
        past_messages = self.database.get_messages_by_user(user_id)

        # Build message history
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        for msg in past_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": update.message.text})

        try:
            # Generate response
            response = await self.current_provider.generate_response(messages)
            
            # Save messages to database
            self.database.save_message(
                user_id=user_id,
                provider=self.current_provider.__class__.__name__,
                model=self.current_provider.model,
                content=update.message.text,
                role="user"
            )
            self.database.save_message(
                user_id=user_id,
                provider=self.current_provider.__class__.__name__,
                model=self.current_provider.model,
                content=response,
                role="assistant"
            )

            # Get message count from history
            message_history = self.database.get_messages_by_user(user_id)
            
            # Create status footer
            status_footer = (
                f"\n\n---\n"
                f"Provider: {self.current_provider.__class__.__name__.replace('Provider', '')}\n"
                f"Model: {self.current_provider.model}\n"
                f"History: {len(message_history)} messages\n"
                f"Conversation: {active_conv['title'] if active_conv else 'New Conversation'}"
            )

            await update.message.reply_text(f"{response}{status_footer}")

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            await update.message.reply_text(
                "Sorry, I encountered an error. Please try again later."
            )

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors in conversations."""
        logger.error(f"Update {update} caused error {context.error}")
        error_message = (
            "Sorry, something went wrong processing your request. "
            "Please try again later."
        )
        if update.effective_message:
            await update.effective_message.reply_text(error_message)

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""
        user_id = update.effective_user.id
        
        try:
            stats = self.database.get_user_stats(user_id)
            
            # Format providers stats
            providers_text = "\n".join(
                f"• {provider}: {count} messages"
                for provider, count in stats['messages_by_provider'].items()
            ) or "No messages yet"

            # Format time periods
            time_stats = stats['time_periods']
            
            stats_message = (
                "📊 *Your Chat Statistics*\n\n"
                f"*Total Messages:* {stats['total_messages']}\n"
                f"• You: {stats['user_messages']} messages\n"
                f"• Bot: {stats['bot_messages']} messages\n\n"
                f"*Messages by Provider:*\n{providers_text}\n\n"
                f"*Most Used Model:* {stats['most_used_model'] or 'None'}\n\n"
                f"*Recent Activity:*\n"
                f"• Today: {time_stats['today']} messages\n"
                f"• This week: {time_stats['this_week']} messages\n"
                f"• This month: {time_stats['this_month']} messages"
            )

            await update.message.reply_text(
                stats_message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Error generating stats for user {user_id}: {str(e)}")
            await update.message.reply_text(
                "Sorry, I encountered an error while generating your stats. Please try again later."
            )

    async def list_conversations_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /conversations command."""
        user_id = update.effective_user.id
        page = context.user_data.get('conv_page', 0)  # Get current page, default to 0
        per_page = 5
        
        conversations = self.database.get_user_conversations(user_id, limit=per_page + 1, offset=page * per_page)
        active_conv = self.database.get_active_conversation(user_id)
        
        if not conversations:
            await update.message.reply_text("You don't have any conversations yet.")
            return
        
        # Check if there are more conversations
        has_next = len(conversations) > per_page
        conversations = conversations[:per_page]  # Limit to per_page items
        
        message = "Your conversations:\n\n"
        for conv in conversations:
            active_marker = "🟢 " if conv['is_active'] else "⚪️ "
            message += f"{active_marker}{conv['title']} ({conv['message_count']} messages)\n"
            message += f"Created: {conv['created_at']}\n\n"
        
        keyboard = []
        for conv in conversations:
            keyboard.append([InlineKeyboardButton(
                f"{'✓ ' if conv['is_active'] else ''}{conv['title']}",
                callback_data=f"conv_{conv['conversation_id']}"
            )])
        
        # Add navigation buttons
        nav_buttons = []
        if page > 0:
            nav_buttons.append(InlineKeyboardButton("Newer", callback_data="conv_page_prev"))
        if has_next:
            nav_buttons.append(InlineKeyboardButton("Older", callback_data="conv_page_next"))
        if nav_buttons:
            keyboard.append(nav_buttons)
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(message, reply_markup=reply_markup)

    async def rename_conversation_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /rename_conversation command."""
        user_id = update.effective_user.id
        active_conv = self.database.get_active_conversation(user_id)
        
        if not active_conv:
            await update.message.reply_text(
                "No active conversation found. Start a new conversation first with /new"
            )
            return
        
        # Set state to wait for new name
        context.user_data['awaiting_rename'] = active_conv['conversation_id']
        
        await update.message.reply_text(
            f"Current conversation name: {active_conv['title']}\n"
            "Please send me the new name for this conversation."
        )

# --- Main ---
def main():
    """Main function to start the bot."""
    # Load configuration
    config = BotConfig.from_env()
    
    # Initialize database
    database = Database(config.db_path)
    
    # Initialize bot with database and config
    bot = TelegramBot(database, config)
    
    try:
        print(f"Starting bot in {config.environment} mode...")
        if config.environment == "production" and config.webhook_url:
            bot.application.run_webhook(
                listen="0.0.0.0",
                port=8443,
                url_path=config.token,
                webhook_url=config.webhook_url
            )
        else:
            bot.application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
        raise

if __name__ == "__main__":
    main()