module.exports = {
  // Project settings
  project: {
    name: "PoseC",
    description: "Pose estimation and analysis project"
  },
  
  // Task settings
  tasks: {
    defaultPriority: "medium",
    priorities: ["low", "medium", "high"],
    statuses: ["pending", "in-progress", "completed", "blocked"]
  },
  
  // Integration settings
  integrations: {
    git: true,
    notifications: true
  },
  
  // Custom fields for tasks
  customFields: {
    "type": ["feature", "bug", "documentation", "research"],
    "estimated_time": "number"
  }
}; 