import React from 'react';
import { useNavigate } from 'react-router-dom';

const CustomNavigate = ({ to }) => {
  const navigate = useNavigate();
  React.useEffect(() => {
    navigate(to);
  }, [navigate, to]);
  return null;
};

export default CustomNavigate;
