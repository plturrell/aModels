/**
 * LocalAI Module - Apple-Level Chat
 * Premium message bubbles, smooth animations, delightful UX
 */

import React, { useState, useRef, useEffect } from "react";
import {
  Box,
  Typography,
  TextField,
  Button,
  Container,
  alpha,
  Fade,
  Grow
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import SmartToyIcon from "@mui/icons-material/SmartToy";
import { PremiumCard } from "../../components/PremiumCard";

interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export function LocalAIModule() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [sending, setSending] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { 
      role: "user", 
      content: input,
      timestamp: new Date()
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setSending(true);

    // Simulate AI response with typing delay
    setTimeout(() => {
      const aiMessage: Message = {
        role: "assistant",
        content: "I'm a simulated AI response. In production, I would connect to your LocalAI models and provide intelligent, context-aware responses based on your input.",
        timestamp: new Date()
      };
      setMessages((prev) => [...prev, aiMessage]);
      setSending(false);
    }, 1500);
  };

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', bgcolor: '#F5F5F7' }}>
      <Container maxWidth="md" sx={{ flex: 1, display: 'flex', flexDirection: 'column', py: 4 }}>
        {/* Header */}
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
            Chat
          </Typography>
          <Typography variant="body1" sx={{ color: 'text.secondary' }}>
            Powered by LocalAI models
          </Typography>
        </Box>

        {/* Messages Container */}
        <Box 
          sx={{ 
            flex: 1, 
            overflow: 'auto',
            mb: 3,
            '&::-webkit-scrollbar': {
              width: '8px',
            },
            '&::-webkit-scrollbar-thumb': {
              background: alpha('#000', 0.1),
              borderRadius: '4px',
            },
          }}
        >
          {messages.length === 0 && (
            <Fade in timeout={600}>
              <Box 
                sx={{ 
                  textAlign: "center", 
                  py: 12,
                  animation: 'fadeIn 0.6s ease-out',
                }}
              >
                <Box
                  sx={{
                    width: 80,
                    height: 80,
                    borderRadius: 4,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: `linear-gradient(135deg, ${alpha('#007AFF', 0.1)} 0%, ${alpha('#5856D6', 0.1)} 100%)`,
                    mx: 'auto',
                    mb: 3,
                  }}
                >
                  <SmartToyIcon sx={{ fontSize: 40, color: 'primary.main' }} />
                </Box>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                  Start a conversation
                </Typography>
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  Ask me anything. I'm here to help.
                </Typography>
              </Box>
            </Fade>
          )}

          {messages.map((msg, index) => (
            <Grow in key={index} timeout={400}>
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                  mb: 2,
                  animation: 'slideIn 0.3s ease-out',
                  '@keyframes slideIn': {
                    from: {
                      opacity: 0,
                      transform: msg.role === 'user' ? 'translateX(20px)' : 'translateX(-20px)',
                    },
                    to: {
                      opacity: 1,
                      transform: 'translateX(0)',
                    },
                  },
                }}
              >
                <PremiumCard
                  sx={{
                    maxWidth: '75%',
                    background: msg.role === 'user' 
                      ? 'linear-gradient(135deg, #007AFF 0%, #5AC8FA 100%)'
                      : '#FFFFFF',
                    color: msg.role === 'user' ? '#FFFFFF' : 'text.primary',
                    border: msg.role === 'assistant' ? `1px solid ${alpha('#000', 0.06)}` : 'none',
                    boxShadow: msg.role === 'user'
                      ? '0px 8px 24px rgba(0, 122, 255, 0.25)'
                      : '0px 4px 20px rgba(0, 0, 0, 0.06)',
                  }}
                  hoverable={false}
                >
                  <Box sx={{ p: 2.5 }}>
                    <Typography 
                      variant="caption" 
                      sx={{ 
                        opacity: 0.7,
                        display: 'block',
                        mb: 1,
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        fontSize: '0.7rem',
                        letterSpacing: '0.5px'
                      }}
                    >
                      {msg.role === "user" ? "You" : "Assistant"}
                    </Typography>
                    <Typography 
                      variant="body1" 
                      sx={{ 
                        lineHeight: 1.6,
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word'
                      }}
                    >
                      {msg.content}
                    </Typography>
                  </Box>
                </PremiumCard>
              </Box>
            </Grow>
          ))}

          {sending && (
            <Fade in>
              <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
                <PremiumCard sx={{ background: '#FFFFFF' }} hoverable={false}>
                  <Box sx={{ p: 2.5, display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Box
                      sx={{
                        display: 'flex',
                        gap: 0.75,
                        '& span': {
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          bgcolor: 'primary.main',
                          opacity: 0.6,
                          animation: 'pulse 1.4s ease-in-out infinite',
                        },
                        '& span:nth-of-type(2)': {
                          animationDelay: '0.2s',
                        },
                        '& span:nth-of-type(3)': {
                          animationDelay: '0.4s',
                        },
                        '@keyframes pulse': {
                          '0%, 80%, 100%': {
                            transform: 'scale(0.8)',
                            opacity: 0.5,
                          },
                          '40%': {
                            transform: 'scale(1.2)',
                            opacity: 1,
                          },
                        },
                      }}
                    >
                      <span />
                      <span />
                      <span />
                    </Box>
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                      Thinking...
                    </Typography>
                  </Box>
                </PremiumCard>
              </Box>
            </Fade>
          )}
          
          <div ref={messagesEndRef} />
        </Box>

        {/* Input Area */}
        <PremiumCard glass sx={{ p: 2 }}>
          <Box sx={{ display: "flex", gap: 2, alignItems: 'center' }}>
            <TextField
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              fullWidth
              multiline
              maxRows={4}
              variant="standard"
              onKeyPress={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              InputProps={{
                disableUnderline: true,
              }}
              sx={{
                '& .MuiInputBase-root': {
                  fontSize: '1rem',
                  lineHeight: 1.5,
                },
              }}
            />
            <Button
              variant="contained"
              onClick={handleSend}
              disabled={sending || !input.trim()}
              sx={{
                minWidth: 48,
                minHeight: 48,
                borderRadius: '50%',
                p: 0,
                background: 'linear-gradient(135deg, #007AFF 0%, #5AC8FA 100%)',
                boxShadow: '0px 4px 12px rgba(0, 122, 255, 0.3)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #0051D5 0%, #007AFF 100%)',
                  boxShadow: '0px 6px 16px rgba(0, 122, 255, 0.4)',
                },
                '&:disabled': {
                  background: alpha('#007AFF', 0.3),
                },
              }}
            >
              <SendIcon />
            </Button>
          </Box>
        </PremiumCard>
      </Container>
    </Box>
  );
}
