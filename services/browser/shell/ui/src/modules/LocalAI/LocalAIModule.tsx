/**
 * LocalAI Module - Apple-Level Chat
 * Premium message bubbles, smooth animations, delightful UX
 * Now connected to real LocalAI API
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
  Grow,
  Alert,
  Chip,
  Link
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import SmartToyIcon from "@mui/icons-material/SmartToy";
import { PremiumCard } from "../../components/dashboard/PremiumCard";
import { useLocalAIChatStore } from "../../state/useLocalAIChatStore";
import { useServiceContext } from "../../hooks/useServiceContext";

export function LocalAIModule() {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const {
    messages,
    followUps,
    pending,
    error,
    model,
    sendMessage,
    applyFollowUp,
    reset
  } = useLocalAIChatStore();
  
  const { getContextForLocalAI, setLocalAIContext } = useServiceContext();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, pending]);

  const handleSend = async () => {
    if (!input.trim() || pending) return;
    
    const message = input.trim();
    setInput("");
    
    // Get cross-service context and include it in the message
    const context = getContextForLocalAI();
    const enhancedMessage = context && context !== "No context available."
      ? `${message}\n\nContext from other services:\n${context}`
      : message;
    
    // Update LocalAI context
    setLocalAIContext({
      lastMessage: message,
      context: context
    });
    
    await sendMessage(enhancedMessage);
  };

  const handleFollowUp = async (prompt: string) => {
    await applyFollowUp(prompt);
  };

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', bgcolor: '#F5F5F7' }}>
      <Container maxWidth="md" sx={{ flex: 1, display: 'flex', flexDirection: 'column', py: 4 }}>
        {/* Header */}
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
            Chat
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
            <Typography variant="body1" sx={{ color: 'text.secondary' }}>
              Powered by LocalAI
            </Typography>
            {model && (
              <Chip 
                label={model} 
                size="small" 
                sx={{ 
                  height: 20, 
                  fontSize: '0.65rem',
                  bgcolor: alpha('#007AFF', 0.1),
                  color: 'primary.main'
                }} 
              />
            )}
          </Box>
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

          {messages.map((msg) => (
            <Grow in key={msg.id} timeout={400}>
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
                      : msg.error
                      ? alpha('#f44336', 0.1)
                      : '#FFFFFF',
                    color: msg.role === 'user' ? '#FFFFFF' : msg.error ? 'error.main' : 'text.primary',
                    border: msg.role === 'assistant' ? `1px solid ${alpha('#000', 0.06)}` : 'none',
                    boxShadow: msg.role === 'user'
                      ? '0px 8px 24px rgba(0, 122, 255, 0.25)'
                      : '0px 4px 20px rgba(0, 0, 0, 0.06)',
                  }}
                  hoverable={false}
                >
                  <Box sx={{ p: 2.5 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography 
                        variant="caption" 
                        sx={{ 
                          opacity: 0.7,
                          fontWeight: 600,
                          textTransform: 'uppercase',
                          fontSize: '0.7rem',
                          letterSpacing: '0.5px'
                        }}
                      >
                        {msg.role === "user" ? "You" : "Assistant"}
                      </Typography>
                      {msg.streaming && (
                        <Chip 
                          label="Streaming" 
                          size="small" 
                          sx={{ 
                            height: 18, 
                            fontSize: '0.65rem',
                            opacity: 0.7
                          }} 
                        />
                      )}
                    </Box>
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
                    {msg.citations && msg.citations.length > 0 && (
                      <Box sx={{ mt: 2, pt: 2, borderTop: `1px solid ${alpha('#000', 0.1)}` }}>
                        <Typography variant="caption" sx={{ fontWeight: 600, mb: 1, display: 'block' }}>
                          Sources:
                        </Typography>
                        {msg.citations.map((citation, idx) => (
                          <Link
                            key={citation.id || idx}
                            href={citation.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            sx={{
                              display: 'block',
                              fontSize: '0.75rem',
                              color: 'primary.main',
                              mb: 0.5,
                              textDecoration: 'none',
                              '&:hover': { textDecoration: 'underline' }
                            }}
                          >
                            {idx + 1}. {citation.label}
                          </Link>
                        ))}
                      </Box>
                    )}
                  </Box>
                </PremiumCard>
              </Box>
            </Grow>
          ))}

          {pending && (
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

          {error && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => useLocalAIChatStore.getState().error = null}>
              {error}
            </Alert>
          )}

          {followUps.length > 0 && messages.length > 0 && !pending && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1, display: 'block' }}>
                Suggested follow-ups:
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {followUps.map((followUp) => (
                  <Chip
                    key={followUp.id}
                    label={followUp.label}
                    onClick={() => handleFollowUp(followUp.prompt)}
                    size="small"
                    sx={{
                      cursor: 'pointer',
                      '&:hover': {
                        bgcolor: 'primary.main',
                        color: 'white'
                      }
                    }}
                  />
                ))}
              </Box>
            </Box>
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
              disabled={pending || !input.trim()}
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
