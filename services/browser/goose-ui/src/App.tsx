import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  List,
  ListItem,
  ListItemText,
  Chip,
  AppBar,
  Toolbar,
  IconButton,
  Divider,
  Card,
  CardContent
} from '@mui/material';
import {
  Send as SendIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Code as CodeIcon,
  Terminal as TerminalIcon
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import axios from 'axios';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  tools?: string[];
}

interface Task {
  id: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: string;
}

const GOOSE_API_URL = process.env.REACT_APP_GOOSE_API || 'http://localhost:8082';

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post(`${GOOSE_API_URL}/chat`, {
        message: input,
        session_id: 'default'
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.data.response,
        timestamp: new Date(),
        tools: response.data.tools_used
      };

      setMessages(prev => [...prev, assistantMessage]);

      if (response.data.task) {
        setTasks(prev => [...prev, {
          id: response.data.task.id,
          description: response.data.task.description,
          status: 'running'
        }]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'system',
        content: 'Error: Could not connect to Goose API. Please ensure the service is running.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setTasks([]);
  };

  const renderMessage = (message: Message) => {
    return (
      <ListItem
        key={message.id}
        sx={{
          flexDirection: 'column',
          alignItems: message.role === 'user' ? 'flex-end' : 'flex-start',
          mb: 2
        }}
      >
        <Paper
          elevation={2}
          sx={{
            p: 2,
            maxWidth: '80%',
            bgcolor: message.role === 'user' ? 'primary.light' : 'background.paper',
            color: message.role === 'user' ? 'primary.contrastText' : 'text.primary'
          }}
        >
          <Typography variant="caption" color="text.secondary" gutterBottom>
            {message.role === 'user' ? 'You' : 'Goose'} • {message.timestamp.toLocaleTimeString()}
          </Typography>
          
          <ReactMarkdown
            components={{
              code({ node, inline, className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || '');
                return !inline && match ? (
                  <SyntaxHighlighter
                    style={vscDarkPlus}
                    language={match[1]}
                    PreTag="div"
                    {...props}
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                ) : (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              }
            }}
          >
            {message.content}
          </ReactMarkdown>

          {message.tools && message.tools.length > 0 && (
            <Box mt={1}>
              {message.tools.map((tool, idx) => (
                <Chip
                  key={idx}
                  label={tool}
                  size="small"
                  icon={<CodeIcon />}
                  sx={{ mr: 0.5 }}
                />
              ))}
            </Box>
          )}
        </Paper>
      </ListItem>
    );
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <AppBar position="static">
        <Toolbar>
          <TerminalIcon sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Goose Web UI
          </Typography>
          <IconButton color="inherit" onClick={clearChat}>
            <DeleteIcon />
          </IconButton>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ flexGrow: 1, display: 'flex', py: 3, gap: 2 }}>
        {/* Chat Area */}
        <Paper sx={{ flex: 3, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
            <Typography variant="h6">Chat</Typography>
          </Box>

          <List sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
            {messages.length === 0 && (
              <Box textAlign="center" py={8}>
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  Welcome to Goose!
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Ask me to help you with development tasks, file operations, or coding.
                </Typography>
              </Box>
            )}
            {messages.map(renderMessage)}
            <div ref={messagesEndRef} />
          </List>

          <Divider />

          <Box sx={{ p: 2, display: 'flex', gap: 1 }}>
            <TextField
              fullWidth
              multiline
              maxRows={4}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder="Ask Goose to help with a task..."
              disabled={loading}
            />
            <Button
              variant="contained"
              endIcon={<SendIcon />}
              onClick={sendMessage}
              disabled={loading || !input.trim()}
            >
              Send
            </Button>
          </Box>
        </Paper>

        {/* Tasks Sidebar */}
        <Paper sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h6">Tasks</Typography>
            <IconButton size="small">
              <RefreshIcon />
            </IconButton>
          </Box>

          <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
            {tasks.length === 0 ? (
              <Typography variant="body2" color="text.secondary" textAlign="center" py={4}>
                No active tasks
              </Typography>
            ) : (
              tasks.map(task => (
                <Card key={task.id} sx={{ mb: 2 }}>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="caption" color="text.secondary">
                        Task {task.id}
                      </Typography>
                      <Chip 
                        label={task.status} 
                        size="small"
                        color={
                          task.status === 'completed' ? 'success' :
                          task.status === 'failed' ? 'error' :
                          task.status === 'running' ? 'primary' : 'default'
                        }
                      />
                    </Box>
                    <Typography variant="body2">{task.description}</Typography>
                    {task.result && (
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                        {task.result}
                      </Typography>
                    )}
                  </CardContent>
                </Card>
              ))
            )}
          </Box>
        </Paper>
      </Container>
    </Box>
  );
}

export default App;
