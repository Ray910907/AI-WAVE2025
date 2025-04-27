// Global variables
    const API_ENDPOINT = 'https://ykc1w19h5f.execute-api.us-west-2.amazonaws.com/test/pred';
    const API_RATE_LIMIT = 500; // Max requests per second
    const PAGE_SIZE = 10; // Items per page
    const DEMO_FILE_NAME = 'validation.csv'; // Demo file name
    
    let accounts = []; // Array to store all analyzed accounts
    let currentPage = 1;
    let isProcessing = false;
    let requestQueue = [];
    let lastRequestTime = 0;
    
    let riskChart = null; // Variable to store the chart instance
    let totalAccountsToProcess = 0;
    let processedAccounts = 0;
    let demoFileSelected = false; // Flag to track if demo file is selected
    
    // DOM Elements
    const analyzeBtn = document.getElementById('fileActionBtn');
    const csvFileInput = document.getElementById('csvFileInput');
    const statusMessages = document.getElementById('statusMessages');
    const accountsList = document.getElementById('accountsList');
    const noData = document.getElementById('noData');
    const tableContainer = document.getElementById('tableContainer');
    const pagination = document.getElementById('pagination');
    const refreshBtn = document.getElementById('refreshBtn');
    const clearLogBtn = document.getElementById('clearLogBtn');
    const analyzedAccountsEl = document.getElementById('analyzedAccounts');
    const riskAccountsEl = document.getElementById('riskAccounts');
    const progressBar = document.getElementById('progressBar');
    const processingStatus = document.getElementById('processingStatus');
    const processingStatusContainer = document.getElementById('processingStatusContainer');
    
    // Initialize the page
    function init() {
      updateStats();
      updateTable();
      initChart();
      
      // Event listeners
      setupFileActionButton();
      //refreshBtn.addEventListener('click', refreshData);
      clearLogBtn.addEventListener('click', clearLogs);
      
      // Add file change listener
      csvFileInput.addEventListener('change', function() {
        demoFileSelected = false; // Reset demo file flag when user selects their own file
        if (csvFileInput.files.length > 0) {
          fileInputName.textContent = csvFileInput.files[0].name;
          analyzeBtn.textContent = '上傳並分析';
        } else {
          fileInputName.textContent = '尚未選擇檔案';
          analyzeBtn.textContent = '選擇檔案';
        }
      });
      
      // Add demo file button
      setupDemoFileButton();
    }
    
    // Setup demo file button
    function setupDemoFileButton() {
      // Create the demo file button
      const demoFileBtn = document.createElement('button');
      demoFileBtn.id = 'demoFileBtn';
      demoFileBtn.textContent = '使用範例檔案';
      
      // Match the style of the main file action button
      demoFileBtn.className = 'batch-analyze';
      demoFileBtn.style.width = 'auto';
      demoFileBtn.style.padding = '8px 15px'; 
      demoFileBtn.style.display = 'inline-block';
      demoFileBtn.style.marginLeft = '10px';
      
      // Insert the button after the analyze button
      analyzeBtn.parentNode.insertBefore(demoFileBtn, analyzeBtn.nextSibling);
      
      // Add click event listener
      demoFileBtn.addEventListener('click', selectDemoFile);
    }
    
    // Select the demo file
    function selectDemoFile() {
      demoFileSelected = true;
      fileInputName.textContent = DEMO_FILE_NAME;
      analyzeBtn.textContent = '分析範例檔案';
      addLogMessage(`已選擇範例檔案: ${DEMO_FILE_NAME}`, 'info');
    }
    
    // Setup the combined file action button
    function setupFileActionButton() {
      analyzeBtn.addEventListener('click', function() {
        if (demoFileSelected) {
          // Handle demo file analysis
          startDemoFileAnalysis();
        } else if (csvFileInput.files.length > 0) {
          // If file is selected, start analysis
          startAnalysis();
        } else {
          // If no file is selected, open file dialog
          csvFileInput.click();
        }
      });
    }
    
    // Start analysis with demo file
    function startDemoFileAnalysis() {
      // Disable button during processing
      analyzeBtn.disabled = true;
      
      // Reset progress and show progress bar
      resetProgress();
      processingStatusContainer.style.display = 'block';
      
      addLogMessage(`開始分析範例檔案: ${DEMO_FILE_NAME}`, 'loading');
      
      // Fetch the demo file
      fetch(`${DEMO_FILE_NAME}`)
        .then(response => {
          if (!response.ok) {
            throw new Error('範例檔案載入失敗');
          }
          return response.text();
        })
        .then(csvText => {
          // Process the CSV text directly
          const lines = csvText.split('\n');
          const headers = lines[0].split(',').map(header => header.trim());
          const acctNbrIndex = headers.findIndex(h => h.toUpperCase() === 'ACCT_NBR');
          
          if (acctNbrIndex === -1) {
            throw new Error('CSV 檔案中必須包含 ACCT_NBR 欄位');
          }
          
          const accountsData = [];
          
          // Process each line (skip header)
          for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line) {
              const values = line.split(',');
              const accountId = values[acctNbrIndex];
              const dataValues = [...values];
              dataValues.splice(acctNbrIndex, 1);
              const dataString = dataValues.join(',');
              
              accountsData.push({
                accountId,
                dataString,
                allFields: values,
                headers: headers
              });
            }
          }
          
          totalAccountsToProcess = accountsData.length;
          // Update the analyzed accounts count to match total accounts being processed
          analyzedAccountsEl.textContent = totalAccountsToProcess;
          addLogMessage(`範例檔案解析成功，開始處理 ${totalAccountsToProcess} 個帳戶`, 'success');
          
          // Queue up all accounts for processing
          requestQueue = [...accountsData];
          
          // Update the initial progress
          updateProgressBar();
          
          // Start processing the queue if not already processing
          if (!isProcessing) {
            processQueue();
          }
        })
        .catch(error => {
          addLogMessage(`範例檔案處理失敗: ${error.message}`, 'error');
          analyzeBtn.disabled = false;
          processingStatusContainer.style.display = 'none';
        });
    }
    
    // Add a log message to the status panel
    function addLogMessage(message, type = 'info') {
      const messageDiv = document.createElement('div');
      messageDiv.className = `status-message status-${type}`;
      
      // If type is loading, add a spinner
      if (type === 'loading') {
        const spinner = document.createElement('span');
        spinner.className = 'loading-spinner';
        messageDiv.appendChild(spinner);
      }
      
      const now = new Date();
      const timeString = now.toLocaleTimeString();
      
      messageDiv.appendChild(document.createTextNode(`[${timeString}] ${message}`));
      statusMessages.appendChild(messageDiv);
      statusMessages.scrollTop = statusMessages.scrollHeight;
    }
    
    // Clear all log messages
    function clearLogs() {
      statusMessages.innerHTML = '';
      addLogMessage('日誌已清除', 'success');
    }
    
    // Update statistics
    function updateStats() {
      const totalAccounts = accounts.length;
      const riskAccounts = accounts.filter(acc => acc.riskScore >= 0.7).length;
      
      // Only update the analyzed accounts count if it's smaller than total processed
      // This prevents the count from going backwards during processing
      if (parseInt(analyzedAccountsEl.textContent) < totalAccounts) {
        analyzedAccountsEl.textContent = totalAccounts;
      }
      
      riskAccountsEl.textContent = riskAccounts;
      
      // Update the chart whenever stats are updated
      updateChart();
    }
    
    // Get risk level based on score
    function getRiskLevel(score) {
      if (score >= 0.75) return { level: '高風險', class: 'risk-high' };
      if (score >= 0.5) return { level: '中風險', class: 'risk-medium' };
      return { level: '低風險', class: 'risk-low' };
    }
    
    // Parse CSV file and extract data
    function parseCSV(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event) => {
          try {
            const csvText = event.target.result;
            const lines = csvText.split('\n');
            
            // Get headers (first line)
            const headers = lines[0].split(',').map(header => header.trim());
            
            // Find the index of ACCT_NBR field
            const acctNbrIndex = headers.findIndex(h => h.toUpperCase() === 'ACCT_NBR');
            
            if (acctNbrIndex === -1) {
              reject(new Error('CSV 檔案中必須包含 ACCT_NBR 欄位'));
              return;
            }
            
            const accounts = [];
            
            // Process each line (skip header)
            for (let i = 1; i < lines.length; i++) {
              const line = lines[i].trim();
              if (line) {
                const values = line.split(',');
                
                // Extract account number
                const accountId = values[acctNbrIndex];
                
                // Create a copy of the values and remove the ACCT_NBR
                const dataValues = [...values];
                dataValues.splice(acctNbrIndex, 1);
                
                // Join the remaining values
                const dataString = dataValues.join(',');
                
                accounts.push({
                  accountId,
                  dataString,
                  allFields: values, // Store all fields for display
                  headers: headers  // Keep headers for reference
                });
              }
            }
            
            resolve(accounts);
          } catch (error) {
            reject(error);
          }
        };
        
        reader.onerror = () => {
          reject(new Error('讀取檔案時發生錯誤'));
        };
        
        reader.readAsText(file);
      });
    }
    
    // Update progress bar
    function updateProgressBar() {
      const percentComplete = (processedAccounts / totalAccountsToProcess) * 100;
      progressBar.style.width = `${percentComplete}%`;
      progressBar.textContent = `${Math.round(percentComplete)}%`;
      processingStatus.textContent = `處理中: ${processedAccounts} / ${totalAccountsToProcess}`;
    }
    
    // Reset progress tracking
    function resetProgress() {
      processedAccounts = 0;
      totalAccountsToProcess = 0;
      progressBar.style.width = '0%';
      progressBar.textContent = '0%';
      processingStatusContainer.style.display = 'none';
      demoFileSelected = false; // Reset demo file flag when starting new analysis
    }
    
    // Update the accounts table
    function updateTable(filteredAccounts = null) {
      const displayAccounts = accounts; // Always display all accounts
      
      // Show/hide no data message
      if (displayAccounts.length === 0) {
        noData.style.display = 'block';
        accountsList.innerHTML = '';
        pagination.innerHTML = '';
        return;
      }
      
      noData.style.display = 'none';
      
      // Calculate pagination
      const totalPages = Math.ceil(displayAccounts.length / PAGE_SIZE);
      const startIndex = (currentPage - 1) * PAGE_SIZE;
      const endIndex = Math.min(startIndex + PAGE_SIZE, displayAccounts.length);
      
      // Sort accounts by risk score (highest first)
      const sortedAccounts = [...displayAccounts].sort((a, b) => b.riskScore - a.riskScore);
      
      // Clear existing table rows
      accountsList.innerHTML = '';
      
      // Add new rows
      for (let i = startIndex; i < endIndex; i++) {
        const account = sortedAccounts[i];
        const risk = getRiskLevel(account.riskScore);
        
        const row = document.createElement('tr');
        
        // Account ID
        const idCell = document.createElement('td');
        idCell.textContent = account.accountId;
        row.appendChild(idCell);
        
        // Risk Score
        const scoreCell = document.createElement('td');
        scoreCell.textContent = account.riskScore.toFixed(4);
        row.appendChild(scoreCell);
        
        // Risk Level
        const riskCell = document.createElement('td');
        const riskSpan = document.createElement('span');
        riskSpan.className = `risk-indicator ${risk.class}`;
        riskSpan.textContent = risk.level;
        riskCell.appendChild(riskSpan);
        row.appendChild(riskCell);
        
        // Details button
        const detailsCell = document.createElement('td');
        const detailsBtn = document.createElement('button');
        detailsBtn.textContent = '詳細資料';
        detailsBtn.style.padding = '5px 10px';
        detailsBtn.style.backgroundColor = '#00539F';
        detailsBtn.style.color = 'white';
        detailsBtn.style.border = 'none';
        detailsBtn.style.borderRadius = '4px';
        detailsBtn.style.cursor = 'pointer';
        
        detailsBtn.addEventListener('click', () => {
          showAccountDetails(account);
        });
        
        detailsCell.appendChild(detailsBtn);
        row.appendChild(detailsCell);
        
        accountsList.appendChild(row);
      }
      
      // Update pagination
      updatePagination(totalPages);
    }
    
    // Show account details in a modal
    function showAccountDetails(account) {
      // Create modal backdrop
      const modalBackdrop = document.createElement('div');
      modalBackdrop.style.position = 'fixed';
      modalBackdrop.style.top = '0';
      modalBackdrop.style.left = '0';
      modalBackdrop.style.width = '100%';
      modalBackdrop.style.height = '100%';
      modalBackdrop.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
      modalBackdrop.style.zIndex = '1000';
      modalBackdrop.style.display = 'flex';
      modalBackdrop.style.justifyContent = 'center';
      modalBackdrop.style.alignItems = 'center';
      
      // Create modal content
      const modalContent = document.createElement('div');
      modalContent.style.backgroundColor = 'white';
      modalContent.style.padding = '20px';
      modalContent.style.borderRadius = '8px';
      modalContent.style.maxWidth = '80%';
      modalContent.style.maxHeight = '80%';
      modalContent.style.overflow = 'auto';
      modalContent.style.position = 'relative';
      
      // Create close button
      const closeButton = document.createElement('button');
      closeButton.textContent = 'X';
      closeButton.style.position = 'absolute';
      closeButton.style.top = '10px';
      closeButton.style.right = '10px';
      closeButton.style.border = 'none';
      closeButton.style.backgroundColor = '#E60012';
      closeButton.style.color = 'white';
      closeButton.style.borderRadius = '50%';
      closeButton.style.width = '30px';
      closeButton.style.height = '30px';
      closeButton.style.cursor = 'pointer';
      closeButton.style.fontSize = '14px';
      closeButton.addEventListener('click', () => {
        document.body.removeChild(modalBackdrop);
      });
      
      // Create header
      const header = document.createElement('h3');
      header.textContent = `帳號 ${account.accountId} 詳細資料`;
      header.style.marginBottom = '20px';
      header.style.color = '#00539F';
      
      // Create risk score display
      const riskDiv = document.createElement('div');
      const risk = getRiskLevel(account.riskScore);
      riskDiv.innerHTML = `<strong>風險分數: </strong>${account.riskScore.toFixed(4)} <span class="risk-indicator ${risk.class}" style="margin-left: 10px;">${risk.level}</span>`;
      riskDiv.style.marginBottom = '20px';
      riskDiv.style.fontSize = '16px';
      
      // Create table for all fields
      const table = document.createElement('table');
      table.style.width = '100%';
      table.style.borderCollapse = 'collapse';
      
      // Add table header
      const thead = document.createElement('thead');
      const headerRow = document.createElement('tr');
      
      const fieldHeader = document.createElement('th');
      fieldHeader.textContent = '欄位名稱';
      fieldHeader.style.padding = '10px';
      fieldHeader.style.backgroundColor = '#f5f5f5';
      fieldHeader.style.borderBottom = '1px solid #e0e0e0';
      headerRow.appendChild(fieldHeader);
      
      const valueHeader = document.createElement('th');
      valueHeader.textContent = '欄位值';
      valueHeader.style.padding = '10px';
      valueHeader.style.backgroundColor = '#f5f5f5';
      valueHeader.style.borderBottom = '1px solid #e0e0e0';
      headerRow.appendChild(valueHeader);
      
      thead.appendChild(headerRow);
      table.appendChild(thead);
      
      // Add table body with all fields
      const tbody = document.createElement('tbody');
      
      if (account.headers && account.allFields) {
        account.headers.forEach((header, index) => {
          const row = document.createElement('tr');
          
          const fieldCell = document.createElement('td');
          fieldCell.textContent = header;
          fieldCell.style.padding = '10px';
          fieldCell.style.borderBottom = '1px solid #e0e0e0';
          row.appendChild(fieldCell);
          
          const valueCell = document.createElement('td');
          valueCell.textContent = account.allFields[index] || '';
          valueCell.style.padding = '10px';
          valueCell.style.borderBottom = '1px solid #e0e0e0';
          row.appendChild(valueCell);
          
          tbody.appendChild(row);
        });
      }
      
      table.appendChild(tbody);
      
      // Assemble modal
      modalContent.appendChild(closeButton);
      modalContent.appendChild(header);
      modalContent.appendChild(riskDiv);
      modalContent.appendChild(table);
      modalBackdrop.appendChild(modalContent);
      
      // Add to body
      document.body.appendChild(modalBackdrop);
    }
    
    // Update pagination controls
    function updatePagination(totalPages) {
      pagination.innerHTML = '';
      
      if (totalPages <= 1) return;
      
      // Previous button
      if (currentPage > 1) {
        const prevBtn = document.createElement('button');
        prevBtn.textContent = '←';
        prevBtn.addEventListener('click', () => {
          currentPage--;
          updateTable();
        });
        pagination.appendChild(prevBtn);
      }
      
      // Page numbers
      const startPage = Math.max(1, currentPage - 2);
      const endPage = Math.min(totalPages, currentPage + 2);
      
      for (let i = startPage; i <= endPage; i++) {
        const pageBtn = document.createElement('button');
        pageBtn.textContent = i;
        if (i === currentPage) {
          pageBtn.className = 'active';
        }
        pageBtn.addEventListener('click', () => {
          currentPage = i;
          updateTable();
        });
        pagination.appendChild(pageBtn);
      }
      
      // Next button
      if (currentPage < totalPages) {
        const nextBtn = document.createElement('button');
        nextBtn.textContent = '→';
        nextBtn.addEventListener('click', () => {
          currentPage++;
          updateTable();
        });
        pagination.appendChild(nextBtn);
      }
    }
    
    // Start the analysis process
    function startAnalysis() {
      const fileInput = csvFileInput;
      
      if (!fileInput.files || fileInput.files.length === 0) {
        addLogMessage('請選擇 CSV 檔案', 'error');
        return;
      }
      
      const file = fileInput.files[0];
      
      if (!file.name.toLowerCase().endsWith('.csv')) {
        addLogMessage('請選擇有效的 CSV 檔案', 'error');
        return;
      }
      
      // Reset progress and show progress bar
      resetProgress();
      processingStatusContainer.style.display = 'block';
      
      // Disable button during processing
      analyzeBtn.disabled = true;
      
      addLogMessage(`開始分析 CSV 檔案: ${file.name}`, 'loading');
      
      // Parse CSV file
      parseCSV(file)
        .then(accountsData => {
          totalAccountsToProcess = accountsData.length;
          // Update the analyzed accounts count to match total accounts being processed
          analyzedAccountsEl.textContent = totalAccountsToProcess;
          addLogMessage(`CSV 檔案解析成功，開始處理 ${totalAccountsToProcess} 個帳戶`, 'success');
          
          // Queue up all accounts for processing
          requestQueue = [...accountsData];
          
          // Update the initial progress
          updateProgressBar();
          
          // Start processing the queue if not already processing
          if (!isProcessing) {
            processQueue();
          }
        })
        .catch(error => {
          addLogMessage(`CSV 檔案解析失敗: ${error.message}`, 'error');
          analyzeBtn.disabled = false;
          processingStatusContainer.style.display = 'none';
        });
    }
    
    // Process the request queue with rate limiting
    function processQueue() {
      if (requestQueue.length === 0) {
        isProcessing = false;
        analyzeBtn.disabled = false;
        if (demoFileSelected) {
          analyzeBtn.textContent = '分析範例檔案';
        } else {
          analyzeBtn.textContent = '選擇檔案';
        }
        addLogMessage('所有帳戶分析完成', 'success');
        
        // Update the analyzed accounts count to match the total processed
        // before updating stats to prevent potential race conditions
        analyzedAccountsEl.textContent = totalAccountsToProcess;
        
        // Update other stats but preserve the account count
        const currentCount = analyzedAccountsEl.textContent;
        updateStats();
        analyzedAccountsEl.textContent = currentCount;
        
        updateTable();
        // Hide progress bar after completion
        setTimeout(() => {
          processingStatusContainer.style.display = 'none';
        }, 3000);
        return;
      }
      
      isProcessing = true;
      
      // Calculate time to wait for rate limiting
      const now = Date.now();
      const timeToWait = Math.max(0, 1000 / API_RATE_LIMIT - (now - lastRequestTime));
      
      setTimeout(() => {
        const accountData = requestQueue.shift();
        analyzeAccount(accountData);
        lastRequestTime = Date.now();
        processQueue();
      }, timeToWait);
    }
    
    // Analyze a single account
    async function analyzeAccount(accountData) {
      try {
        // Make API call to the AWS API Gateway endpoint with the required format
        await fetch(API_ENDPOINT, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
            // Add Authorization if needed
          },
          body: JSON.stringify({
            data: accountData.dataString
          })
        })
        .then(response => {
          if (!response.ok) {
            throw new Error(`API 回應錯誤: ${response.status}`);
          }
          return response.json();
        })
        .then(result => {
          // Store the result
          const processedAccount = {
            accountId: accountData.accountId,
            riskScore: parseFloat(result.riskScore),
            dataString: accountData.dataString,
            allFields: accountData.allFields,
            headers: accountData.headers
          };

          // Check if this account already exists
          const existingIndex = accounts.findIndex(acc => acc.accountId === accountData.accountId);
          if (existingIndex !== -1) {
            accounts[existingIndex] = processedAccount;
          } else {
            accounts.push(processedAccount);
          }

          // Update processed count and progress bar
          processedAccounts++;
          updateProgressBar();

          // Update UI periodically to avoid performance issues
          if (processedAccounts % 10 === 0 || requestQueue.length === 0) {
            updateStats(); // This will also update the chart
          }
          
          if (requestQueue.length === 0) {
            updateTable();
          }
        });
      } catch (error) {
        // Just increment counter and progress even for failed accounts
        processedAccounts++;
        updateProgressBar();
        
        // Log the error to console instead of UI to reduce clutter
        console.error(`處理帳戶 ${accountData.accountId} 時發生錯誤:`, error);
      }
    }
    
    // Initialize the risk distribution chart
    function initChart() {
      const ctx = document.getElementById('riskDistributionChart').getContext('2d');
      
      riskChart = new Chart(ctx, {
        type: 'pie',
        data: {
          labels: ['高風險', '中風險', '低風險'],
          datasets: [{
            data: [0, 0, 0],
            backgroundColor: [
              '#F44336', // High risk - Red
              '#FFC107', // Medium risk - Yellow
              '#4CAF50'  // Low risk - Green
            ],
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: true,
              text: '帳戶風險分佈圖',
              font: {
                size: 16,
                weight: 'bold'
              }
            },
            legend: {
              position: 'bottom' 
            }
          }
        }
      });
    }
    
    // Update chart data
    function updateChart() {
      if (!riskChart) return;
      
      // Count accounts by risk level
      const highRisk = accounts.filter(acc => acc.riskScore >= 0.75).length;
      const mediumRisk = accounts.filter(acc => acc.riskScore >= 0.5 && acc.riskScore < 0.75).length;
      const lowRisk = accounts.filter(acc => acc.riskScore < 0.5).length;
      
      // Update chart data
      riskChart.data.datasets[0].data = [highRisk, mediumRisk, lowRisk];
      
      // Add counts to labels if there are any accounts
      if (accounts.length > 0) {
        riskChart.data.labels = [
          `高風險 (${highRisk})`, 
          `中風險 (${mediumRisk})`, 
          `低風險 (${lowRisk})`
        ];
      }
      
      riskChart.update();
    }
    
    // Initialize the page
    document.addEventListener('DOMContentLoaded', () => {
      init();
      resetProgress(); // Initialize progress system
    });