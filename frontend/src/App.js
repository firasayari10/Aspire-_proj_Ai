import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import HomePage from './components/HomePage';
import ImageDropzone from './components/ImageDropzone';
import BackButton from './components/BackButton';
import DroneImageDropzone from './components/DroneImageDropzone';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/images" element={
            <>
              <BackButton />
              <main className="App-main">
                <ImageDropzone />
              </main>
            </>
          } />
          <Route path="/drone" element={
            <>
              <BackButton />
              <main className="App-main">
                <DroneImageDropzone />
              </main>
            </>
          } />
        </Routes>
      </div>
    </Router>
  );
}

export default App; 