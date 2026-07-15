module.exports = {
  apps: [
    {
      name: 'vmi-autocommit',
      cwd: '/home/user/webapp',
      script: './scanner/autocommit.sh',
      interpreter: 'bash',
      autorestart: true,
      watch: false,
      instances: 1,
      exec_mode: 'fork'
    },
    {
      name: 'vmi-streamlit',
      cwd: '/home/user/webapp',
      script: 'python3',
      args: '-m streamlit run scanner/webapp_ui.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false',
      autorestart: true,
      watch: false,
      instances: 1,
      exec_mode: 'fork'
    }
  ]
}
