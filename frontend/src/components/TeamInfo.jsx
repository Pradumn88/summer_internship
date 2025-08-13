import React from 'react';

function TeamInfo({ darkMode }) {
  const teamMembers = [
    {
      name: "Pradumn Pandey",
      role: "AI Developer & Project Lead",
      linkedin: "https://www.linkedin.com/in/pradumnpandey/"
    },
    {
      name: "Hemank Kumar",
      role: "Machine Learning Engineer",
      linkedin: "https://www.linkedin.com/in/hemankkumar/"
    },
    {
      name: "Yuvraj Singh Parmar",
      role: "Frontend Developer",
      linkedin: "https://www.linkedin.com/in/yuvraj-singh-parmar-517762326/"
    },
    {
      name: "Arpan Pareek",
      role: "Medical Consultant",
      linkedin: "https://www.linkedin.com/in/arpan-pareek-86582a2b7/"
    },
    {
      name: "Faizah",
      role: "Backend Developer",
      linkedin: "https://www.linkedin.com/in/faaizah-qadri-2a3191311/"
    }
  ];

  return (
    <div className="py-4">
      <h2 className="text-2xl font-bold mb-6 flex items-center justify-center">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
        </svg>
        Development Team
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {teamMembers.map((member, index) => (
          <div 
            key={index} 
            className={`p-4 rounded-xl flex items-center transition-all duration-300 ${
              darkMode 
                ? 'bg-gray-800 hover:bg-gray-700 shadow-lg' 
                : 'bg-white hover:bg-gray-50 shadow-md'
            }`}
          >
            {/* Circular profile placeholder */}
            <div className="mr-4 flex-shrink-0">
              <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
                darkMode ? 'bg-gray-700' : 'bg-gray-200'
              }`}>
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  className={`h-8 w-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    strokeWidth={1.5} 
                    d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" 
                  />
                </svg>
              </div>
            </div>
            
            <div className="min-w-0">
              <a 
                href={member.linkedin} 
                target="_blank" 
                rel="noopener noreferrer"
                className={`font-bold text-lg hover:underline flex items-center truncate ${
                  darkMode ? 'text-blue-400' : 'text-blue-600'
                }`}
              >
                <span className="truncate">{member.name}</span>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-2 flex-shrink-0" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
                </svg>
              </a>
              <p className={`mt-1 truncate ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                {member.role}
              </p>
            </div>
          </div>
        ))}
      </div>
      
      <div className={`mt-8 pt-4 border-t text-center ${
        darkMode ? 'border-gray-700 text-gray-500' : 'border-gray-300 text-gray-600'
      }`}>
        <p>Summer Internship Project • August 2025</p>
      </div>
    </div>
  );
}

export default TeamInfo;